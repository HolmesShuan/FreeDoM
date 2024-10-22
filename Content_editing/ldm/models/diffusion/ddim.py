"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from .low_filter import LowFilter, VGG16ContentEncoder, VGG16StyleEncoder

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)

from einops import rearrange
from PIL import Image
import os
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import minimize, Bounds, minimize_scalar

import torchvision

to_tensor = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

from .clip.base_clip import CLIPEncoder


def cagrad(grads, c=0.5):
    # g1 = grads[:,0]
    # g2 = grads[:,1]
    g1 = grads[0].view(-1)
    g2 = grads[1].view(-1)
    g0 = (grads[0] + grads[1]) / 2.0

    g11 = torch.dot(g1, g1).cpu().item()
    g12 = torch.dot(g1, g2).cpu().item()
    g22 = torch.dot(g2, g2).cpu().item()

    g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12 + 1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return (
            coef
            * np.sqrt(x**2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-4)
            + 0.5 * x * (g11 + g22 - 2 * g12)
            + (0.5 + x) * (g12 - g22)
            + g22
        )

    res = minimize_scalar(obj, bounds=(0, 1), method="bounded")
    x = res.x

    gw = x * grads[0] + (1 - x) * grads[1]
    gw_norm = np.sqrt(x**2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-4)

    lmbda = coef / (gw_norm + 1e-4)
    g = g0 + lmbda * gw
    return g / (1 + c)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer(
            "alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            "sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu()))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),
        )

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer(
            "ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps
        )

    # @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        content_ref_img_path=None,
        content_ref_mask_path=None,
        style_ref_img_path=None,
        control_detail=dict(),
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        use_clip_style_loss = (
            control_detail["use_clip_style_loss"]
            if "use_clip_style_loss" in control_detail
            else False
        )
        if use_clip_style_loss:
            self.clip_style_encoder = CLIPEncoder(
                need_ref=True, ref_path=style_ref_img_path
            ).cuda()
        else:
            self.vgg_style_filter = VGG16StyleEncoder(
                need_ref=True,
                ref_path=style_ref_img_path,
            ).cuda()
        self.vgg_content_filter = VGG16ContentEncoder(
            need_ref=True,
            ref_path=content_ref_img_path,
            ref_mask_path=content_ref_mask_path,
        ).cuda()

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            control_detail=control_detail,
        )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        control_detail=dict(),
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(
            time_range, desc="DDIM Sampler", total=total_steps, disable=False
        )

        count1 = 0

        for i, step in enumerate(iterator):

            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                tt = x0[-1 - i].to(device)
                # img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img * mask + (1.0 - mask) * tt

            outs = self.p_sample_ddim_magic_dual_conditional(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                i=i if mask is not None else None,
                control_detail=control_detail,
            )

            img, pred_x0 = outs

            count1 += 1
            # print(pred_x0.size())
            pred_x0_temp = self.model.decode_first_stage(pred_x0)
            pred_x0_temp = torch.clamp((pred_x0_temp + 1.0) / 2.0, min=0.0, max=1.0)
            pred_x0_temp = pred_x0_temp.cpu().permute(0, 2, 3, 1).detach().numpy()
            pred_x0_torch = torch.from_numpy(pred_x0_temp).permute(0, 3, 1, 2)
            count2 = 0
            for x_sample in pred_x0_torch:
                count2 += 1
                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                img_save = Image.fromarray(x_sample.astype(np.uint8))
                img_save.save(
                    os.path.join(
                        "/home/hexiangyu/FreeDoM/Content_editing/intermediates",
                        "{}_{}.png".format(count1, count2),
                    )
                )

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            #     intermediates['x_inter'].append(img)
            #     intermediates['pred_x0'].append(pred_x0)
            intermediates["x_inter"].append(img)
            intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    def p_sample_ddim_magic_conditional(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        i=None,
        control_detail=dict(),
    ):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        # hyperparameters:
        # "repeat" is the number for time-travel strategy;
        # "start" and "end" are the end points for the range of guidance;
        content_start = (
            control_detail["content_start_steps"]
            if "content_start_steps" in control_detail
            else 20
        )
        content_end = (
            control_detail["content_end_steps"]
            if "content_end_steps" in control_detail
            else 0
        )
        style_start = (
            control_detail["style_start_steps"]
            if "style_start_steps" in control_detail
            else 20
        )
        style_end = (
            control_detail["style_end_steps"]
            if "style_end_steps" in control_detail
            else 0
        )
        repeat = control_detail["repeat"] if "repeat" in control_detail else 5
        time_reverse_step = (
            control_detail["time_reverse_step"]
            if "time_reverse_step" in control_detail
            else False
        )
        rho_scale = (
            control_detail["rho_scale"] if "rho_scale" in control_detail else 1.0
        )

        for j in range(repeat):
            x = x.detach().requires_grad_(True)

            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat(
                    [unconditional_conditioning[i].expand(*c.shape).to(device), c]
                )
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                correction = e_t - e_t_uncond
                e_t = e_t_uncond + unconditional_guidance_scale * correction

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            alphas = (
                self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            )
            alphas_prev = (
                self.model.alphas_cumprod_prev
                if use_original_steps
                else self.ddim_alphas_prev
            )
            sqrt_one_minus_alphas = (
                self.model.sqrt_one_minus_alphas_cumprod
                if use_original_steps
                else self.ddim_sqrt_one_minus_alphas
            )
            sigmas = (
                self.model.ddim_sigmas_for_original_num_steps
                if use_original_steps
                else self.ddim_sigmas
            )
            sigmas = self.ddim_sigmas

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # if start > index >= end:
            #     D_x0_t = self.model.decode_first_stage(pred_x0)
            #     residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
            #     norm = torch.linalg.norm(residual)

            #     norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            #     rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale / (norm_grad * norm_grad).mean().sqrt().item() * 0.2

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            # calculate x0|x,c1
            x_prev_given_c1 = x_prev.clone()  # .detach()
            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t_given_c1 = self.model.apply_model(x_prev_given_c1, t, c)
            else:
                x_in = torch.cat([x_prev_given_c1] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat(
                    [unconditional_conditioning[i].expand(*c.shape).to(device), c]
                )
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                # unconditional_guidance_scale = 5.0
                correction = e_t - e_t_uncond
                e_t_given_c1 = e_t_uncond + unconditional_guidance_scale * correction

            pred_x0_given_c1 = (
                x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1
            ) / a_t.sqrt()

            if content_start > index >= content_end:
                D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
                residual = self.vgg_content_filter.get_content_residual(D_x0_t)
                style_loss = torch.linalg.norm(residual)
                print("[Debug] Style Loss : ", style_loss.cpu().item(), flush=True)
                style_loss_grad = torch.autograd.grad(
                    outputs=style_loss, inputs=x_prev_given_c1
                )[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                style_loss_grad_1d = style_loss_grad.view(-1)
                style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / style_loss_grad_1d_norm.item()
                gradient_prod = torch.dot(
                    x_prev.view(-1), rho * style_loss_grad_1d.detach().view(-1)
                )
                style_loss_magic_grad = torch.autograd.grad(
                    outputs=gradient_prod, inputs=x
                )[0]
                x_prev = x_prev - style_loss_magic_grad.detach() * rho_scale
            # else:
            #     x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)
            if time_reverse_step:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(
                    x.shape, device, repeat_noise
                )

        return x_prev.detach(), pred_x0.detach()

    # def p_sample_ddim_magic_dual_conditional(
    #     self,
    #     x,
    #     c,
    #     t,
    #     index,
    #     repeat_noise=False,
    #     use_original_steps=False,
    #     quantize_denoised=False,
    #     temperature=1.0,
    #     noise_dropout=0.0,
    #     score_corrector=None,
    #     corrector_kwargs=None,
    #     unconditional_guidance_scale=1.0,
    #     unconditional_conditioning=None,
    #     i=None,
    #     control_detail=dict(),
    # ):
    #     b, *_, device = *x.shape, x.device

    #     x.requires_grad = True
    #     self.model.requires_grad_(True)

    #     # hyperparameters:
    #     # "repeat" is the number for time-travel strategy;
    #     # "start" and "end" are the end points for the range of guidance;
    #     content_start = (
    #         control_detail["content_start_steps"]
    #         if "content_start_steps" in control_detail
    #         else 20
    #     )
    #     content_end = (
    #         control_detail["content_end_steps"]
    #         if "content_end_steps" in control_detail
    #         else 0
    #     )
    #     style_start = (
    #         control_detail["style_start_steps"]
    #         if "style_start_steps" in control_detail
    #         else 20
    #     )
    #     style_end = (
    #         control_detail["style_end_steps"]
    #         if "style_end_steps" in control_detail
    #         else 0
    #     )
    #     repeat = control_detail["repeat"] if "repeat" in control_detail else 5
    #     time_reverse_step = (
    #         control_detail["time_reverse_step"]
    #         if "time_reverse_step" in control_detail
    #         else False
    #     )
    #     content_rho_scale = (
    #         control_detail["content_rho_scale"]
    #         if "content_rho_scale" in control_detail
    #         else 1.0
    #     )
    #     style_rho_scale = (
    #         control_detail["style_rho_scale"]
    #         if "style_rho_scale" in control_detail
    #         else 1.0
    #     )
    #     use_cagrad = (
    #         control_detail["use_cagrad"] if "use_cagrad" in control_detail else False
    #     )
    #     cagrad_weight = (
    #         control_detail["cagrad_weight"]
    #         if "cagrad_weight" in control_detail
    #         else 1.0
    #     )
    #     use_clip_style_loss = (
    #         control_detail["use_clip_style_loss"]
    #         if "use_clip_style_loss" in control_detail
    #         else False
    #     )

    #     for j in range(repeat):
    #         x = x.detach().requires_grad_(True)
    #         if style_start > index >= style_end:
    #             x_2nd = x.clone()

    #         if (
    #             unconditional_conditioning is None
    #             or unconditional_guidance_scale == 1.0
    #         ):
    #             if style_start > index >= style_end:
    #                 e_t_2nd = self.model.apply_model(x_2nd, t, c)
    #             e_t = self.model.apply_model(x, t, c)
    #         else:
    #             x_in = torch.cat([x] * 2)
    #             t_in = torch.cat([t] * 2)
    #             c_in = (
    #                 torch.cat(
    #                     [unconditional_conditioning[i].expand(*c.shape).to(device), c]
    #                 )
    #                 if i is not None
    #                 else torch.cat([unconditional_conditioning, c])
    #             )
    #             e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    #             correction = e_t - e_t_uncond
    #             e_t = e_t_uncond + unconditional_guidance_scale * correction

    #             if style_start > index >= style_end:
    #                 x_in_2nd = torch.cat([x_2nd] * 2)
    #                 t_in = torch.cat([t] * 2)
    #                 c_in = (
    #                     torch.cat(
    #                         [
    #                             unconditional_conditioning[i]
    #                             .expand(*c.shape)
    #                             .to(device),
    #                             c,
    #                         ]
    #                     )
    #                     if i is not None
    #                     else torch.cat([unconditional_conditioning, c])
    #                 )
    #                 e_t_uncond_2nd, e_t_2nd = self.model.apply_model(
    #                     x_in_2nd, t_in, c_in
    #                 ).chunk(2)
    #                 correction_2nd = e_t_2nd - e_t_uncond_2nd
    #                 e_t_2nd = (
    #                     e_t_uncond_2nd + unconditional_guidance_scale * correction_2nd
    #                 )

    #         if score_corrector is not None:
    #             assert self.model.parameterization == "eps"
    #             e_t = score_corrector.modify_score(
    #                 self.model, e_t, x, t, c, **corrector_kwargs
    #             )

    #         alphas = (
    #             self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
    #         )
    #         alphas_prev = (
    #             self.model.alphas_cumprod_prev
    #             if use_original_steps
    #             else self.ddim_alphas_prev
    #         )
    #         sqrt_one_minus_alphas = (
    #             self.model.sqrt_one_minus_alphas_cumprod
    #             if use_original_steps
    #             else self.ddim_sqrt_one_minus_alphas
    #         )
    #         sigmas = (
    #             self.model.ddim_sigmas_for_original_num_steps
    #             if use_original_steps
    #             else self.ddim_sigmas
    #         )
    #         sigmas = self.ddim_sigmas

    #         a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    #         a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    #         beta_t = a_t / a_prev
    #         sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    #         sqrt_one_minus_at = torch.full(
    #             (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
    #         )

    #         # current prediction for x_0
    #         pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    #         c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
    #         c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
    #         c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
    #         c3 = (c3.log() * 0.5).exp()
    #         x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

    #         if style_start > index >= style_end:
    #             pred_x0_2nd = (x_2nd - sqrt_one_minus_at * e_t_2nd) / a_t.sqrt()
    #             x_prev_2nd = (
    #                 c1 * pred_x0_2nd + c2 * x_2nd + c3 * torch.randn_like(pred_x0)
    #             )
    #             # calculate x0|x,c1
    #             x_prev_given_c1_2nd = x_prev_2nd.clone()  # .detach()

    #         if content_start > index >= content_end:
    #             # calculate x0|x,c1
    #             x_prev_given_c1 = x_prev.clone()  # .detach()
    #             if (
    #                 unconditional_conditioning is None
    #                 or unconditional_guidance_scale == 1.0
    #             ):
    #                 e_t_given_c1 = self.model.apply_model(x_prev_given_c1, t, c)
    #             else:
    #                 x_in = torch.cat([x_prev_given_c1] * 2)
    #                 t_in = torch.cat([t] * 2)
    #                 c_in = (
    #                     torch.cat(
    #                         [
    #                             unconditional_conditioning[i]
    #                             .expand(*c.shape)
    #                             .to(device),
    #                             c,
    #                         ]
    #                     )
    #                     if i is not None
    #                     else torch.cat([unconditional_conditioning, c])
    #                 )
    #                 e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    #                 # unconditional_guidance_scale = 5.0
    #                 correction = e_t - e_t_uncond
    #                 e_t_given_c1 = (
    #                     e_t_uncond + unconditional_guidance_scale * correction
    #                 )

    #             pred_x0_given_c1 = (
    #                 x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1
    #             ) / a_t.sqrt()

    #             D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
    #             residual = self.vgg_content_filter.get_content_residual(D_x0_t)
    #             content_loss = torch.linalg.norm(residual)
    #             print("[Debug] Content Loss : ", content_loss.cpu().item(), flush=True)
    #             content_loss_grad = torch.autograd.grad(
    #                 outputs=content_loss, inputs=x_prev_given_c1
    #             )[0]
    #             correction_1d = correction.view(-1)
    #             correction_1d_l2_norm = torch.norm(correction_1d, p=2)
    #             content_loss_grad_1d = content_loss_grad.view(-1)
    #             content_loss_grad_1d_norm = torch.norm(content_loss_grad_1d, p=2)
    #             rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
    #             rho = rho / content_loss_grad_1d_norm.item()
    #             gradient_prod = torch.dot(
    #                 x_prev.view(-1), rho * content_loss_grad_1d.detach().view(-1)
    #             )
    #             content_loss_magic_grad = torch.autograd.grad(
    #                 outputs=gradient_prod, inputs=x
    #             )[0]
    #             x_prev = x_prev - content_loss_magic_grad.detach() * content_rho_scale

    #         if style_start > index >= style_end:
    #             if (
    #                 unconditional_conditioning is None
    #                 or unconditional_guidance_scale == 1.0
    #             ):
    #                 e_t_given_c1_2nd = self.model.apply_model(x_prev_given_c1_2nd, t, c)
    #             else:
    #                 x_in = torch.cat([x_prev_given_c1_2nd] * 2)
    #                 t_in = torch.cat([t] * 2)
    #                 c_in = (
    #                     torch.cat(
    #                         [
    #                             unconditional_conditioning[i]
    #                             .expand(*c.shape)
    #                             .to(device),
    #                             c,
    #                         ]
    #                     )
    #                     if i is not None
    #                     else torch.cat([unconditional_conditioning, c])
    #                 )
    #                 e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
    #                 # unconditional_guidance_scale = 5.0
    #                 correction = e_t - e_t_uncond
    #                 e_t_given_c1_2nd = (
    #                     e_t_uncond + unconditional_guidance_scale * correction
    #                 )

    #             pred_x0_given_c1_2nd = (
    #                 x_prev_given_c1_2nd - sqrt_one_minus_at * e_t_given_c1_2nd
    #             ) / a_t.sqrt()

    #             D_x0_t_2nd = self.model.decode_first_stage(pred_x0_given_c1_2nd)
    #             if use_clip_style_loss:
    #                 residual = self.clip_style_encoder.get_gram_matrix_residual(
    #                     D_x0_t_2nd
    #                 )
    #                 style_loss = torch.linalg.norm(residual)
    #             else:
    #                 style_loss = self.vgg_style_filter.get_gram_matrix_style_loss(
    #                     D_x0_t_2nd
    #                 )
    #             print("[Debug] Style Loss : ", style_loss.cpu().item(), flush=True)
    #             style_loss_grad = torch.autograd.grad(
    #                 outputs=style_loss, inputs=x_prev_given_c1_2nd
    #             )[0]
    #             correction_1d = correction.view(-1)
    #             correction_1d_l2_norm = torch.norm(correction_1d, p=2)
    #             style_loss_grad_1d = style_loss_grad.view(-1)
    #             style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
    #             rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
    #             rho = rho / style_loss_grad_1d_norm.item()
    #             gradient_prod = torch.dot(
    #                 x_prev_2nd.view(-1), rho * style_loss_grad_1d.detach().view(-1)
    #             )
    #             style_loss_magic_grad = torch.autograd.grad(
    #                 outputs=gradient_prod, inputs=x_2nd
    #             )[0]
    #             x_prev = x_prev - style_loss_magic_grad.detach() * style_rho_scale

    #             if use_cagrad:
    #                 grad_0 = content_loss_magic_grad.detach() * content_rho_scale
    #                 grad_1 = style_loss_magic_grad.detach() * style_rho_scale
    #                 cagrad_grad = cagrad([grad_0, grad_1]) * cagrad_weight
    #                 x_prev = x_prev - cagrad_grad + grad_0 + grad_1

    #         if content_start > index >= content_end or style_start > index >= style_end:
    #             if time_reverse_step:
    #                 x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(
    #                     x.shape, device, repeat_noise
    #                 )
    #         else:
    #             x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(
    #                 x.shape, device, repeat_noise
    #             )

    #     return x_prev.detach(), pred_x0.detach()

    def p_sample_ddim_magic_dual_conditional(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        i=None,
        control_detail=dict(),
    ):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        # hyperparameters:
        # "repeat" is the number for time-travel strategy;
        # "start" and "end" are the end points for the range of guidance;
        content_start = (
            control_detail["content_start_steps"]
            if "content_start_steps" in control_detail
            else 20
        )
        content_end = (
            control_detail["content_end_steps"]
            if "content_end_steps" in control_detail
            else 0
        )
        style_start = (
            control_detail["style_start_steps"]
            if "style_start_steps" in control_detail
            else 20
        )
        style_end = (
            control_detail["style_end_steps"]
            if "style_end_steps" in control_detail
            else 0
        )
        repeat = control_detail["repeat"] if "repeat" in control_detail else 5
        time_reverse_step = (
            control_detail["time_reverse_step"]
            if "time_reverse_step" in control_detail
            else False
        )
        content_rho_scale = (
            control_detail["content_rho_scale"]
            if "content_rho_scale" in control_detail
            else 1.0
        )
        style_rho_scale = (
            control_detail["style_rho_scale"]
            if "style_rho_scale" in control_detail
            else 1.0
        )
        use_cagrad = (
            control_detail["use_cagrad"] if "use_cagrad" in control_detail else False
        )
        cagrad_weight = (
            control_detail["cagrad_weight"]
            if "cagrad_weight" in control_detail
            else 1.0
        )
        use_clip_style_loss = (
            control_detail["use_clip_style_loss"]
            if "use_clip_style_loss" in control_detail
            else False
        )

        for j in range(repeat):
            x = x.detach().requires_grad_(True)

            if (
                unconditional_conditioning is None
                or unconditional_guidance_scale == 1.0
            ):
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = (
                    torch.cat(
                        [unconditional_conditioning[i].expand(*c.shape).to(device), c]
                    )
                    if i is not None
                    else torch.cat([unconditional_conditioning, c])
                )
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                correction = e_t - e_t_uncond
                e_t = e_t_uncond + unconditional_guidance_scale * correction

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            alphas = (
                self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            )
            alphas_prev = (
                self.model.alphas_cumprod_prev
                if use_original_steps
                else self.ddim_alphas_prev
            )
            sqrt_one_minus_alphas = (
                self.model.sqrt_one_minus_alphas_cumprod
                if use_original_steps
                else self.ddim_sqrt_one_minus_alphas
            )
            sigmas = (
                self.model.ddim_sigmas_for_original_num_steps
                if use_original_steps
                else self.ddim_sigmas
            )
            sigmas = self.ddim_sigmas

            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if content_start > index >= content_end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                residual = self.vgg_content_filter.get_content_residual(D_x0_t)
                content_loss = torch.linalg.norm(residual)
                print("[Debug] Content Loss : ", content_loss.cpu().item())
                content_loss_grad = torch.autograd.grad(
                    outputs=content_loss, inputs=x, retain_graph=True
                )[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                content_loss_grad_1d = content_loss_grad.view(-1)
                content_loss_grad_1d_norm = torch.norm(content_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / content_loss_grad_1d_norm.item()
                x_prev = x_prev - content_loss_grad * content_rho_scale

            if style_start > index >= style_end:
                # calculate x0|x,c1
                x_prev_given_c1 = x_prev.clone()  # .detach()
                if (
                    unconditional_conditioning is None
                    or unconditional_guidance_scale == 1.0
                ):
                    e_t_given_c1 = self.model.apply_model(x_prev_given_c1, t, c)
                else:
                    x_in = torch.cat([x_prev_given_c1] * 2)
                    t_in = torch.cat([t] * 2)
                    c_in = (
                        torch.cat(
                            [
                                unconditional_conditioning[i]
                                .expand(*c.shape)
                                .to(device),
                                c,
                            ]
                        )
                        if i is not None
                        else torch.cat([unconditional_conditioning, c])
                    )
                    e_t_uncond_2nd, e_t_2nd = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    # unconditional_guidance_scale = 5.0
                    correction_2nd = e_t_2nd - e_t_uncond_2nd
                    e_t_given_c1 = (
                        e_t_uncond_2nd + unconditional_guidance_scale * correction_2nd
                    )

                pred_x0_given_c1 = (
                    x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1
                ) / a_t.sqrt()

                D_x0_t_style = self.model.decode_first_stage(pred_x0_given_c1)
                if use_clip_style_loss:
                    residual = self.clip_style_encoder.get_gram_matrix_residual(
                        D_x0_t_style
                    )
                    style_loss = torch.linalg.norm(residual)
                else:
                    style_loss = self.vgg_style_filter.get_gram_matrix_style_loss(
                        D_x0_t_style
                    )
                print("[Debug] Style Loss : ", style_loss.cpu().item())
                style_loss_grad = torch.autograd.grad(
                    outputs=style_loss, inputs=x_prev_given_c1
                )[0]
                style_loss_grad_1d = style_loss_grad.view(-1)
                style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / style_loss_grad_1d_norm.item()
                gradient_prod = torch.dot(
                    x_prev.view(-1), rho * style_loss_grad_1d.detach().view(-1)
                )
                style_loss_magic_grad = torch.autograd.grad(
                    outputs=gradient_prod, inputs=x, allow_unused=False
                )[0]
                x_prev = x_prev - style_loss_magic_grad.detach() * style_rho_scale

            if content_start > index >= content_end or style_start > index >= style_end:
                if time_reverse_step:
                    x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(
                        x.shape, device, repeat_noise
                    )
            else:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(
                    x.shape, device, repeat_noise
                )

        return x_prev.detach(), pred_x0.detach()

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):

        timesteps = (
            np.arange(self.ddpm_num_timesteps)
            if use_original_steps
            else self.ddim_timesteps
        )
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long
            )
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec
