"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from einops import rearrange
from PIL import Image
import clip
import os
import torch.nn.functional as F
from einops import rearrange

import torchvision
to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

from .clip.base_clip import CLIPEncoder
from .low_filter import LowFilter, VGG16Encoder

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466 * 2 - 1, 0.4578275 * 2 - 1, 0.40821073 * 2 - 1),
            (0.26862954 * 2, 0.26130258 * 2, 0.27577711 * 2),
        )

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    # @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               style_ref_img_path=None,
               control_detail=dict(),
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        if control_detail['use_clip_style_loss']:
            self.image_encoder = CLIPEncoder(need_ref=True, ref_path=style_ref_img_path).cuda()
        else:
            self.image_encoder = VGG16Encoder(need_ref=True, ref_path=style_ref_img_path).cuda()
        self.text_instruction_following_verifer = CLIPEncoder(need_ref=False).cuda()

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    control_detail=control_detail
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      control_detail=dict()):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=False)

        count1 = 0

        for i, step in enumerate(iterator):

            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if 'use_magic' not in control_detail or control_detail['use_magic'] == False:
                sampling_strategy = self.p_sample_ddim_conditional
            else:
                sampling_strategy = self.p_sample_ddim_magic_conditional
            outs = sampling_strategy(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    control_detail=control_detail)
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
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img_save = Image.fromarray(x_sample.astype(np.uint8))
                img_save.save(os.path.join("/home/hexiangyu/FreeDoM/SD_style/intermediates", "{}_{}.png".format(count1, count2)))

            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            # if index % log_every_t == 0 or index == total_steps - 1:
            #     intermediates['x_inter'].append(img)
            #     intermediates['pred_x0'].append(pred_x0)
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    
    # @torch.no_grad()
    def p_sample_ddim_conditional(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      control_detail=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        # hyperparameters: 
        # "repeat" is the number for time-travel strategy; 
        # "start" and "end" are the end points for the range of guidance;
        if index >= 70:
            repeat = 1
        elif 70 > index >= 40:
            repeat = 3
        else:
            repeat = 1 
        start = 70
        end = 30
        
        use_clip_style_loss = (
            control_detail["use_clip_style_loss"] if "use_clip_style_loss" in control_detail else True
        )
        prompt = (
            control_detail["prompt"] if "prompt" in control_detail else None
        )

        for j in range(repeat):

            x = x.detach().requires_grad_(True)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                # unconditional_guidance_scale = 5.0
                correction = e_t - e_t_uncond
                e_t = e_t_uncond + unconditional_guidance_scale * correction

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            sigmas = self.ddim_sigmas
 
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                if use_clip_style_loss:
                    residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                    style_loss = torch.linalg.norm(residual)
                else:
                    style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
                norm_grad = torch.autograd.grad(outputs=style_loss, inputs=x)[0]
                rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale / (norm_grad * norm_grad).mean().sqrt().item() * 0.2


            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if start > index >= end:
                x_prev = x_prev - rho * norm_grad.detach()

            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)
        
        with torch.no_grad():
            D_x0_t = self.model.decode_first_stage(x_prev)
            if use_clip_style_loss:
                residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                style_loss = torch.linalg.norm(residual)
            else:
                style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
            print('[INFO] Style Loss : ', style_loss.cpu().item(), flush=True)
            if prompt is not None:
                cos_sim, _ = self.text_instruction_following_verifer.forward(D_x0_t, prompt)
                # probs = cos_sim.softmax(dim=-1).cpu()
                print('[INFO] Text Loss : ', 1. - cos_sim.item())
        return x_prev.detach(), pred_x0.detach()
    
    
    def p_sample_ddim_magic_conditional(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      control_detail=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)
        
        start = (
            control_detail["style_start_steps"]
            if "style_start_steps" in control_detail
            else 20
        )
        end = (
            control_detail["style_end_steps"]
            if "style_end_steps" in control_detail
            else 0
        )
        repeat = control_detail["repeat"] if "repeat" in control_detail else 1
        time_reverse_step = (
            control_detail["time_reverse_step"]
            if "time_reverse_step" in control_detail
            else False
        )
        style_rho_scale = (
            control_detail["style_rho_scale"] if "style_rho_scale" in control_detail else 1.0
        )
        use_clip_style_loss = (
            control_detail["use_clip_style_loss"] if "use_clip_style_loss" in control_detail else True
        )
        prompt = (
            control_detail["prompt"] if "prompt" in control_detail else None
        )

        # hyperparameters: 
        # "repeat" is the number for time-travel strategy; 
        # "start" and "end" are the end points for the range of guidance;

        for j in range(repeat):
            x = x.detach().requires_grad_(True)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                correction = e_t - e_t_uncond
                e_t = e_t_uncond + unconditional_guidance_scale * correction

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            sigmas = self.ddim_sigmas
 
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            # calculate x0|x,c1
            x_prev_given_c1 = x_prev.clone() # .detach()
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t_given_c1 = self.model.apply_model(x_prev_given_c1, t, c)
            else:
                x_in = torch.cat([x_prev_given_c1] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                # unconditional_guidance_scale = 5.0
                correction = e_t - e_t_uncond
                e_t_given_c1 = e_t_uncond + unconditional_guidance_scale * correction
            pred_x0_given_c1 = (x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1) / a_t.sqrt()
            
            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
                if use_clip_style_loss:
                    residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                    style_loss = torch.linalg.norm(residual)
                else:
                    style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
                style_loss_grad = torch.autograd.grad(outputs=style_loss, inputs=x_prev_given_c1)[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                style_loss_grad_1d = style_loss_grad.view(-1)
                style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / style_loss_grad_1d_norm.item()
                gradient_prod = torch.dot(x_prev.view(-1), rho * style_loss_grad_1d.detach().view(-1))
                style_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=x)[0]
                x_prev = x_prev - style_loss_magic_grad.detach() * style_rho_scale
            # else:
            #     x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

            if time_reverse_step:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        with torch.no_grad():
            D_x0_t = self.model.decode_first_stage(x_prev)
            if use_clip_style_loss:
                residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                style_loss = torch.linalg.norm(residual)
            else:
                style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
            print('[INFO] Style Loss : ', style_loss.cpu().item(), flush=True)
            if prompt is not None:
                cos_sim, _ = self.text_instruction_following_verifer.forward(D_x0_t, prompt)
                print('[INFO] Text Loss : ', 1. - cos_sim.item())
            
        return x_prev.detach(), pred_x0.detach()
    
    
    def p_sample_ddim_magic_conditional_v2(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        # hyperparameters: 
        # "repeat" is the number for time-travel strategy; 
        # "start" and "end" are the end points for the range of guidance;
        # start = 100
        # end = 0
        start = 70
        end = 30
        # end = 0
        if index >= start:
            repeat = 1
        elif start > index >= end + 10:
            repeat = 3
            # repeat = 2
            # repeat = 1
        else:
            repeat = 1 

        for j in range(repeat):

            x = x.detach().requires_grad_(True)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                # unconditional_guidance_scale = 5.0
                correction = e_t - e_t_uncond
                e_t = e_t_uncond + unconditional_guidance_scale * correction

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            sigmas = self.ddim_sigmas
 
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

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
            x_prev_given_c1 = x_prev.clone() # .detach()
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t_given_c1 = self.model.apply_model(x_prev_given_c1, t, c)
            else:
                x_in = torch.cat([x_prev_given_c1] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                # unconditional_guidance_scale = 5.0
                correction = e_t - e_t_uncond
                e_t_given_c1 = e_t_uncond + unconditional_guidance_scale * correction
            pred_x0_given_c1 = (x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1) / a_t.sqrt()
            
            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
                # residual = self.image_encoder.get_gram_matrix_residual(D_x0_t)
                # residual = self.low_pass_filter.get_low_fass_filter_residual(D_x0_t)
                residual = self.vgg_content_filter.get_content_residual(D_x0_t)
                style_loss = torch.linalg.norm(residual)
                print('[Debug] Style Loss : ', style_loss.cpu().item())
                style_loss_grad = torch.autograd.grad(outputs=style_loss, inputs=x_prev_given_c1)[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                style_loss_grad_1d = style_loss_grad.view(-1)
                style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                # rho = rho / style_loss_grad_1d_norm.item() * 0.15
                rho = rho / style_loss_grad_1d_norm.item()
                # print('[Debug] rho : ', rho)
                
            if start > index >= end:
                gradient_prod = torch.dot(x_prev.view(-1), rho * style_loss_grad_1d.detach().view(-1))
                style_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=x)[0]
                with torch.no_grad():
                    style_loss_grad_vec = style_loss_magic_grad.flatten()
                    engery_func_grad_vec = e_t.flatten()
                    cos_sim = F.cosine_similarity(style_loss_grad_vec, engery_func_grad_vec, dim=0)
                    # print('[INFO] Cos Sim : ', cos_sim.cpu().item())
                    # if cos_sim < 0:
                    #     x_prev = x_prev - style_loss_magic_grad.detach()
                x_prev = x_prev - style_loss_magic_grad.detach()
            else:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)
            # if start > index >= end:
            #     x_prev = x_prev - rho * norm_grad.detach()

            # x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

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
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec