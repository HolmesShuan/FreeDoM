"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from PIL import Image
import torchvision
import torchvision.utils as vutils
from scipy.optimize import minimize, Bounds, minimize_scalar
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

import cv2
from torchvision.transforms import ToPILImage

to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

from .clip.base_clip import CLIPEncoder, VGG16Encoder

from .arcface.model import IDLoss

from einops import rearrange
from PIL import Image
import os

import dlib
from skimage import transform as trans
from skimage import io


def mgd(grads):
    g1 = grads[0].view(-1)
    g2 = grads[1].view(-1)

    g11 = torch.dot(g1, g1)
    g12 = torch.dot(g1, g2)
    g22 = torch.dot(g2, g2)

    if g12 < min(g11, g22):
        x = (g22 - g12) / (g11 + g22 - 2*g12 + 1e-8)
    elif g11 < g22:
        x = 1
    else:
        x = 0
    print('[Debug] MDGA Scaling Factor : ', x.cpu().item())
    g_mgd = x * grads[0] + (1 - x) * grads[1] # mgd gradient g_mgd
    return g_mgd


def cagrad(grads, c=0.5):
    # g1 = grads[:,0]
    # g2 = grads[:,1]
    g1 = grads[0].view(-1)
    g2 = grads[1].view(-1)
    g0 = (grads[0] + grads[1]) / 2.0
    
    g11 = torch.dot(g1, g1).cpu().item()
    g12 = torch.dot(g1, g2).cpu().item()
    g22 = torch.dot(g2, g2).cpu().item()

    g0_norm = 0.5 * np.sqrt(g11 + g22 + 2*g12 + 1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * grads[0] + (1-x) * grads[1]
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm + 1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)


def get_points_and_rec(img, detector, shape_predictor, size_threshold=999):
    """
        这个函数的作用是检测人脸并获取人脸的五个关键点，同时返回人脸的矩形框位置。
    """
    dets = detector(img, 1)
    if len(dets) == 0:
        return None, None
    
    all_points = []
    rec_list = []
    for det in dets:
        if isinstance(detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold: 
            break
        rec_list.append(rec)
        shape = shape_predictor(img, rec) 
        single_points = []
        for i in range(5):
            single_points.append([shape.part(i).x, shape.part(i).y])
        all_points.append(np.array(single_points))
    if len(all_points) <= 0:
        return None, None
    else:
        return all_points, rec_list


def align_and_save(img, src_points, template_path, template_scale=1, img_size=256):
    """
        这个函数的作用是根据模板对图片中的人脸进行对齐，并返回变换矩阵 M。
        使用 scikit-image 的 SimilarityTransform 计算源点和模板点之间的相似变换矩阵 M。
    """
    out_size = (img_size, img_size)
    reference = np.load(template_path) / template_scale * (img_size / 256)
    M_list = []
    shape_list = []
    
    for idx, spoint in enumerate(src_points):
        tform = trans.SimilarityTransform()
        tform.estimate(spoint, reference)
        M = tform.params[0:2,:]
        M_list.append(M)
        shape_list.append(img.shape)
    return M_list, shape_list


def align_and_save_dir(src_path, template_path='./pretrain_models/FFHQ_template.npy', template_scale=4, use_cnn_detector=True, img_size=256):
    """
        这个函数的作用是加载图片并调用 get_points_and_rec 函数进行人脸检测和对齐操作。
    """
    if use_cnn_detector:
        detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    else:
        detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')

    img_path = src_path
    img = dlib.load_rgb_image(img_path)

    points, rec_list = get_points_and_rec(img, detector, sp)
    if points is not None:
        return align_and_save(img, points, template_path, template_scale, img_size=img_size)


def get_tensor_M(src_path, index=0):
    """
        这个函数的作用是将人脸对齐后的变换矩阵 M 转换为 PyTorch 张量格式，方便后续在神经网络中使用。
    """
    M, s = align_and_save_dir(src_path)
    M = M[index]
    s = s[index]
    h, w = s[0], s[1]
    a = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    Mt = torch.Tensor(
        [  
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )
    Mt[:2, :] = torch.Tensor(M)
    Mt = torch.inverse(Mt)
    h, w = 256, 256
    b = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    b = torch.inverse(b)
    Mt = a.matmul(Mt)
    Mt = Mt.matmul(b)[:2].unsqueeze(0)
    return Mt


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", add_condition_mode="face_id", ref_path=None, face_ref_path=None, face_ref_2nd_path=None, no_freedom=False, image_size=224, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.add_condition_mode = add_condition_mode
        self.no_freedom = no_freedom
        if self.add_condition_mode == "face_id":
            self.idloss = IDLoss(ref_path=face_ref_path, ref_2nd_path=face_ref_2nd_path).cuda()
            M = get_tensor_M(ref_path)  # 获取对齐矩阵 M，描述如何将人脸图片变换为标准对齐的形状。
            self.grid = F.affine_grid(M, (1, 3, 256, 256), align_corners=True).cuda()  # 基于变换矩阵 M 生成用于几何变换的网格
            self.grid_2nd = None
            if 'couple' in ref_path:
                print('[INFO] Group Photo Mode.')
                M_2nd = get_tensor_M(ref_path, index=1)  # 获取对齐矩阵 M，描述如何将人脸图片变换为标准对齐的形状。
                self.grid_2nd = F.affine_grid(M_2nd, (1, 3, 256, 256), align_corners=True).cuda()  # 基于变换矩阵 M 生成用于几何变换的网格
            else:
                print('[INFO] Single Portrait Mode.')
        elif self.add_condition_mode == "style":
            # image_encoder = CLIPEncoder(need_ref=True, ref_path=ref_path).cuda()
            image_encoder = VGG16Encoder(need_ref=True, ref_path=ref_path, image_size=image_size).cuda()
            self.image_encoder = image_encoder.requires_grad_(False)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
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
        self.use_mgda = True

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
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               start_step=100,
               end_step=0,
               lr=0.5,
               detail_control=dict(),
               **kwargs
               ):
        print('[Debug] CLDM/DDIM_HACKED')
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

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
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    start_step=start_step,
                                                    end_step=end_step,
                                                    lr=lr,
                                                    detail_control=detail_control
                                                    )
        return samples, intermediates

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, start_step=100, end_step=0, lr=0.5, detail_control=dict()):
        
        device = self.model.betas.device
        b = shape[0]

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        # time_range_reverse = time_range[::-1]
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps.")
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if self.add_condition_mode == "style":
                if detail_control['use_magic']:
                    sampling_strategy = self.p_sample_ddim_style_magic
                else:
                    sampling_strategy = self.p_sample_ddim_style
                outs = sampling_strategy(img, total_steps, start_step, end_step, lr, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                          quantize_denoised=quantize_denoised, temperature=temperature,
                                          noise_dropout=noise_dropout, score_corrector=score_corrector,
                                          corrector_kwargs=corrector_kwargs,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          dynamic_threshold=dynamic_threshold,
                                          detail_control=detail_control)
            elif self.add_condition_mode == "face_id":
                if detail_control['use_magic']:
                    sampling_strategy = self.p_sample_ddim_pose_magic
                else:
                    sampling_strategy = self.p_sample_ddim_pose
                outs = sampling_strategy(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold,
                                        detail_control=detail_control)
            img, pred_x0 = outs
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)
            
        return img, intermediates

    def p_sample_ddim_style(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, detail_control=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        if 70 > index >= 40:
            repeat = 5
        else:
            repeat = 1 

        start = 70
        end = 30
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
                # print('[Debug] Style Loss : ', style_loss.cpu().item())
                norm_grad = torch.autograd.grad(outputs=style_loss, inputs=x)[0]
                rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale 
                rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * 0.2

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if start > index >= end:
                x_prev = x_prev - rho * norm_grad.detach()
            
            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        with torch.no_grad():
            x0 = self.model.decode_first_stage(x_prev)
            style_loss = self.image_encoder.get_gram_matrix_style_loss(x0)
            print('[Debug] Style Loss : ', style_loss.cpu().item())
        
        return x_prev.detach(), pred_x0.detach()
    
    def p_sample_ddim_style_magic(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, detail_control=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        repeat = detail_control['repeat'] if 'repeat' in detail_control else 1
        start = detail_control['start_steps'] if 'start_steps' in detail_control else 60
        end = detail_control['end_steps'] if 'end_steps' in detail_control else 0
        time_reverse_step = detail_control['time_reverse_step'] if 'time_reverse_step' in detail_control else False
        rho_scale = detail_control['rho_scale'] if 'rho_scale' in detail_control else 0.08
        
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)
             
            # calculate x0|x,c1
            if self.model.parameterization != "v":
                x_prev_given_c1 = x_prev.clone() # .detach()
                e_t_given_c1 = self.model.predict_eps_from_z_and_v(x_prev_given_c1, t, model_output)
                pred_x0_given_c1 = (x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1) / a_t.sqrt()
            else:
                pred_x0_given_c1 = self.model.predict_start_from_z_and_v(x_prev_given_c1, t, model_output)

            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
                style_loss = self.image_encoder.get_gram_matrix_style_loss(D_x0_t)
                # print('[Debug] Style Loss : ', style_loss.cpu().item())
                style_loss_grad = torch.autograd.grad(outputs=style_loss, inputs=x_prev_given_c1)[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                style_loss_grad_1d = style_loss_grad.view(-1)
                style_loss_grad_1d_norm = torch.norm(style_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / style_loss_grad_1d_norm.item()
                gradient_prod = torch.dot(x_prev.view(-1), rho * style_loss_grad_1d.detach().view(-1))
                style_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=x)[0]
                x_prev = x_prev - style_loss_magic_grad.detach() * rho_scale
            
            if time_reverse_step:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)
        
        with torch.no_grad():
            x0 = self.model.decode_first_stage(x_prev)
            style_loss = self.image_encoder.get_gram_matrix_style_loss(x0)
            print('[Debug] Style Loss : ', style_loss.cpu().item())
            
        return x_prev.detach(), pred_x0.detach()
    

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        repeat = 1 
        start = 100
        end = -10
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                # print('[Debug] unconditional_conditioning is not None')
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)
            
        return x_prev.detach(), pred_x0.detach()
    
    def p_sample_ddim_pose(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, detail_control=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        repeat = 1
        start = 40
        end = -10
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
                
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0)
                warp_D_x0_t = F.grid_sample(D_x0_t, self.grid, align_corners=True)
                residual = self.idloss.get_residual(warp_D_x0_t)
                face_id_loss = torch.linalg.norm(residual)
                # print('[INFO] 1st. Face loss : ', face_id_loss.cpu().item())
                if self.grid_2nd is not None:
                    warp_D_x0_t_2nd = F.grid_sample(D_x0_t, self.grid_2nd, align_corners=True)
                    residual_2nd = self.idloss.get_residual(warp_D_x0_t_2nd, dual_sup=True)
                    face_id_loss_2nd = torch.linalg.norm(residual_2nd)
                    face_id_loss_grad = torch.autograd.grad(outputs=face_id_loss+face_id_loss_2nd, inputs=x)[0]
                    # print('[INFO] 2nd. Face loss : ', face_id_loss_2nd.cpu().item())
                else:
                    face_id_loss_grad = torch.autograd.grad(outputs=face_id_loss, inputs=x)[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                face_id_loss_grad_1d = face_id_loss_grad.view(-1)
                face_id_loss_grad_1d_norm = torch.norm(face_id_loss_grad_1d, p=2)
                rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale
                rho = rho / (face_id_loss_grad * face_id_loss_grad).mean().sqrt().item() * 0.08

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)
                
            if start > index >= end:
                x_prev = x_prev - rho * face_id_loss_grad.detach()
            
            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        with torch.no_grad():
            x0 = self.model.decode_first_stage(x_prev)
            warp_D_x0_t = F.grid_sample(x0, self.grid, align_corners=True)
            residual = self.idloss.get_residual(warp_D_x0_t)
            face_id_loss = torch.linalg.norm(residual)
            print('[INFO] 1st. Face loss : ', face_id_loss.cpu().item())
            if self.grid_2nd is not None:
                warp_D_x0_t_2nd = F.grid_sample(x0, self.grid_2nd, align_corners=True)
                residual_2nd = self.idloss.get_residual(warp_D_x0_t_2nd, dual_sup=True)
                face_id_loss_2nd = torch.linalg.norm(residual_2nd)
                print('[INFO] 2nd. Face loss : ', face_id_loss_2nd.cpu().item())
        
        return x_prev.detach(), pred_x0.detach()

    
    def p_sample_ddim_pose_magic(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, detail_control=dict()):
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)

        repeat = detail_control['repeat'] if 'repeat' in detail_control else 1
        start = detail_control['start_steps'] if 'start_steps' in detail_control else 60
        end = detail_control['end_steps'] if 'end_steps' in detail_control else 0
        cagrad_weight = detail_control['cagrad_weight'] if 'cagrad_weight' in detail_control else 0.08
        time_reverse_step = detail_control['time_reverse_step'] if 'time_reverse_step' in detail_control else False
        use_cagrad = detail_control['use_cagrad'] if 'use_cagrad' in detail_control else False
        per_channel_cagrad = detail_control['per_channel_cagrad'] if 'per_channel_cagrad' in detail_control else True
        rho_scale = detail_control['rho_scale'] if 'rho_scale' in detail_control else 0.08
        
        if self.no_freedom:
            start = end = -10

        for j in range(repeat):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                # print('[Debug] unconditional_conditioning is not None')
                model_t = self.model.apply_model(x, t, c)
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output
                
            if self.grid_2nd is not None:
                x_2nd = x.clone()
                e_t_2nd = self.model.predict_eps_from_z_and_v(x_2nd, t, model_output)

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
                
            # print('[Debug] use_original_steps is ', use_original_steps)
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)
                
            if self.grid_2nd is not None:
                if self.model.parameterization != "v":
                    pred_x0_2nd = (x_2nd - sqrt_one_minus_at * e_t_2nd) / a_t.sqrt()
                else:
                    pred_x0_2nd = self.model.predict_start_from_z_and_v(x_2nd, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)
            
            if self.grid_2nd is not None:
                x_prev_2nd = c1 * pred_x0_2nd + c2 * x_2nd + c3 * torch.randn_like(pred_x0)
            
            # calculate x0|x,c1
            if self.model.parameterization != "v":
                x_prev_given_c1 = x_prev.clone() # .detach()
                e_t_given_c1 = self.model.predict_eps_from_z_and_v(x_prev_given_c1, t, model_output)
                pred_x0_given_c1 = (x_prev_given_c1 - sqrt_one_minus_at * e_t_given_c1) / a_t.sqrt()
                if self.grid_2nd is not None:
                    x_prev_given_c1_2nd = x_prev_2nd.clone() # .detach()
                    e_t_given_c1_2nd = self.model.predict_eps_from_z_and_v(x_prev_given_c1_2nd, t, model_output)
                    pred_x0_given_c1_2nd = (x_prev_given_c1_2nd - sqrt_one_minus_at * e_t_given_c1_2nd) / a_t.sqrt()
            else:
                pred_x0_given_c1 = self.model.predict_start_from_z_and_v(x_prev_given_c1, t, model_output)
                if self.grid_2nd is not None:
                    pred_x0_given_c1_2nd = self.model.predict_start_from_z_and_v(x_prev_given_c1, t, model_output)
            
            if start > index >= end:
                D_x0_t = self.model.decode_first_stage(pred_x0_given_c1)
                # print('[Debug] D_x0_t.shape : ', D_x0_t.shape)
                warp_D_x0_t = F.grid_sample(D_x0_t, self.grid, align_corners=True)
                residual = self.idloss.get_residual(warp_D_x0_t)
                face_id_loss = torch.linalg.norm(residual)
                # print('[INFO] 1st. Face loss : ', face_id_loss.cpu().item())
                face_id_loss_grad = torch.autograd.grad(outputs=face_id_loss, inputs=x_prev_given_c1)[0]
                correction_1d = correction.view(-1)
                correction_1d_l2_norm = torch.norm(correction_1d, p=2)
                face_id_loss_grad_1d = face_id_loss_grad.view(-1)
                face_id_loss_grad_1d_norm = torch.norm(face_id_loss_grad_1d, p=2)
                rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                rho = rho / face_id_loss_grad_1d_norm.item()
                gradient_prod = torch.dot(x_prev.view(-1), rho * face_id_loss_grad.detach().view(-1))
                face_id_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=x)[0]
                
                if self.grid_2nd is not None:
                    D_x0_t_2nd = self.model.decode_first_stage(pred_x0_given_c1_2nd)
                    # print('[Debug] D_x0_t_2nd.shape : ', D_x0_t_2nd.shape)
                    warp_D_x0_t_2nd = F.grid_sample(D_x0_t_2nd, self.grid_2nd, align_corners=True)
                    residual_2nd = self.idloss.get_residual(warp_D_x0_t_2nd, dual_sup=True)
                    face_id_loss_2nd = torch.linalg.norm(residual_2nd)
                    # print('[INFO] 2nd. Face loss : ', face_id_loss_2nd.cpu().item())
                    face_id_loss_grad = torch.autograd.grad(outputs=face_id_loss_2nd, inputs=x_prev_given_c1_2nd)[0]
                    face_id_loss_grad_1d = face_id_loss_grad.view(-1)
                    face_id_loss_grad_1d_norm = torch.norm(face_id_loss_grad_1d, p=2)
                    rho = correction_1d_l2_norm.item() * unconditional_guidance_scale
                    rho = rho / face_id_loss_grad_1d_norm.item()
                    gradient_prod_2nd = torch.dot(x_prev_2nd.view(-1), rho * face_id_loss_grad.detach().view(-1))
                    face_id_loss_magic_grad_2nd = torch.autograd.grad(outputs=gradient_prod_2nd, inputs=x_2nd)[0]
                
                    if use_cagrad:
                        grad_0 = face_id_loss_magic_grad.detach()
                        grad_1 = face_id_loss_magic_grad_2nd.detach()
                        if per_channel_cagrad:
                            channel_num = grad_0.size(1)
                            cagrad_grad = []
                            for k in range(channel_num):
                                cagrad_grad.append(cagrad([grad_0[:, k, :, :], grad_1[:, k, :, :]]))
                            cagrad_grad = torch.stack(cagrad_grad, dim=1)
                        else:
                            cagrad_grad = cagrad([grad_0, grad_1])
                        cagrad_grad = cagrad_grad * cagrad_weight
                        x_prev = x_prev - cagrad_grad 
                    else:
                        x_prev = x_prev - (face_id_loss_magic_grad + face_id_loss_magic_grad_2nd) * rho_scale
                else:
                    x_prev = x_prev - face_id_loss_magic_grad * rho_scale
            
            if time_reverse_step:
                x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        with torch.no_grad():
            x0 = self.model.decode_first_stage(x_prev)
            warp_D_x0_t = F.grid_sample(x0, self.grid, align_corners=True)
            residual = self.idloss.get_residual(warp_D_x0_t)
            face_id_loss = torch.linalg.norm(residual)
            print('[INFO] 1st. Face loss : ', face_id_loss.cpu().item())
            if self.grid_2nd is not None:
                warp_D_x0_t_2nd = F.grid_sample(x0, self.grid_2nd, align_corners=True)
                residual_2nd = self.idloss.get_residual(warp_D_x0_t_2nd, dual_sup=True)
                face_id_loss_2nd = torch.linalg.norm(residual_2nd)
                print('[INFO] 2nd. Face loss : ', face_id_loss_2nd.cpu().item())
        
        return x_prev.detach(), pred_x0.detach()

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

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
               use_original_steps=False, callback=None):

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
            if callback: callback(i)
        return x_dec