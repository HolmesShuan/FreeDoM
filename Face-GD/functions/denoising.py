import torch
from tqdm import tqdm
import numpy as np
import torchvision.utils as tvu
import torchvision
import os

from .clip.base_clip import CLIPEncoder
from .face_parsing.model import FaceParseTool
from .anime2sketch.model import FaceSketchTool 
from .landmark.model import FaceLandMarkTool
from .arcface.model import IDLoss
from scipy.optimize import minimize, Bounds, minimize_scalar

import torch.nn.functional as F


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
    print('[Debug] MDGA Scaling Factor : ', x)
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


def cagrad_third_order(grads, alpha=0.5, rescale=1):
    stacked_grads = torch.stack(grads, dim=0)  # Shape: [N, C, H, W]
    reshaped_grads = stacked_grads.view(len(grads), -1).t()  # Shape: [C*H*W, N]
    GG = reshaped_grads.t().mm(reshaped_grads).cpu() # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt() # norm of the average gradient

    x_start = np.ones(3) / 3
    bnds = tuple((0,1) for x in x_start)
    cons=({'type':'eq','fun':lambda x:1-sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()
    def objfn(x):
        return (x.reshape(1,3).dot(A).dot(b.reshape(3, 1)) + c * np.sqrt(x.reshape(1,3).dot(A).dot(x.reshape(3,1))+1e-8)).sum()
    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(reshaped_grads.device)
    gw = (reshaped_grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = reshaped_grads.mean(1) + lmbda * gw
    if rescale== 0:
        return g.view_as(grads[0])
    elif rescale == 1:
        return g.view_as(grads[0]) / (1 + alpha**2)
    else:
        return g.view_as(grads[0]) / (1 + alpha)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def clip_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, prompt=None, stop=100, domain="face"):
    clip_encoder = CLIPEncoder().cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        elif domain == "imagenet":
            if 800 >= i >= 500:
                repeat = 10
            else:
                repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # get guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def parse_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = parser.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et

        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def clip_parse_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=None, prompt=None, stop=100, domain="face", ref_path=None):
    clip_encoder = CLIPEncoder().cuda()
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    # for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            xt_parse = xt.clone() # .detach()
            
            et = model(xt, t)
            et_parse = model(xt_parse, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # text guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            # if not i <= stop:
            #     print('[INFO] text loss : ', norm.cpu().item())
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            # parse gradient
            if not i <= stop:
                x0_t_parse = (xt_parse - et_parse * (1 - at).sqrt()) / at.sqrt()
                parse_residual = parser.get_residual(x0_t_parse)
                parse_norm = torch.linalg.norm(parse_residual)
                # print('[INFO] seg loss : ', parse_norm.cpu().item())
                parse_norm_grad = torch.autograd.grad(outputs=parse_norm, inputs=xt_parse)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            if not i <= stop:
                with torch.no_grad():
                    parse_loss_magic_grad_vec = parse_norm_grad.flatten()
                    engery_func_grad_vec = norm_grad.flatten()
                    cos_sim = F.cosine_similarity(parse_loss_magic_grad_vec, engery_func_grad_vec, dim=0)
                    print('[INFO] Cos Sim : ', cos_sim.cpu().item())
            
            # use guided gradient
            parse_rho = at.sqrt() * rho_scale[1]
            if not i <= stop:
                xt_next -= parse_rho * parse_norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
        if True or not i <= stop:
            with torch.no_grad():
                residual = clip_encoder.get_residual(xt_next, prompt)
                norm = torch.linalg.norm(residual)
                print('[INFO] text loss : ', norm.cpu().item())
                parse_residual = parser.get_residual(xt_next)
                parse_norm = torch.linalg.norm(parse_residual)
                print('[INFO] seg loss : ', parse_norm.cpu().item())

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def clip_parse_ddim_diffusion_magic(x, seq, model, b, cls_fn=None, rho_scale=None, prompt=None, stop=100, domain="face", ref_path=None):
    clip_encoder = CLIPEncoder().cuda()
    parser = FaceParseTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    # for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            
            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # text guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            # if not i <= stop:
            #     print('[INFO] text loss : ', norm.cpu().item())
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt, retain_graph=True)[0]
            
            # if not i <= stop:
            #     with torch.no_grad():
            #         parse_residual = parser.get_residual(x0_t)
            #         parse_norm = torch.linalg.norm(parse_residual)
            #         print('[INFO] seg loss : ', parse_norm.cpu().item())

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad

            # parse gradient
            if not i <= stop:
                # calculate x|c1
                xt_parse = xt_next.clone() # .detach()
                et_parse = model(xt_parse, t)
                
                if et_parse.size(1) == 6:
                    et_parse = et_parse[:, :3]
                
                x0_t_parse = (xt_parse - et_parse * (1 - at).sqrt()) / at.sqrt()
                parse_residual = parser.get_residual(x0_t_parse)
                parse_norm = torch.linalg.norm(parse_residual)
                # print('[INFO] seg loss : ', parse_norm.cpu().item())
                parse_norm_grad = torch.autograd.grad(outputs=parse_norm, inputs=xt_parse)[0]
                gradient_prod = torch.dot(xt_next.view(-1), parse_norm_grad.detach().view(-1))
                parse_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=xt, allow_unused=False)[0]
                
            # use guided gradient
            if not i <= stop:
                parse_rho = at.sqrt() * rho_scale[1]
                xt_next -= parse_rho * parse_loss_magic_grad.detach()
                x0_t = x0_t_parse.detach()
            else:
                x0_t = x0_t.detach()
                
            use_mdga = True
            if use_mdga and not i <= stop:
                # mdga_grad = mgd([rho * norm_grad.detach(), parse_rho * parse_loss_magic_grad.detach()])
                ca_grad = cagrad([rho * norm_grad.detach(), parse_rho * parse_loss_magic_grad.detach()])   
                # xt_next -= mdga_grad + rho * norm_grad.detach() + parse_rho * parse_loss_magic_grad.detach()
                xt_next -= ca_grad + rho * norm_grad.detach() + parse_rho * parse_loss_magic_grad.detach()
                
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
        if True or not i <= stop:
            with torch.no_grad():
                residual = clip_encoder.get_residual(xt_next, prompt)
                norm = torch.linalg.norm(residual)
                print('[INFO] text loss : ', norm.cpu().item())
                parse_residual = parser.get_residual(xt_next)
                parse_norm = torch.linalg.norm(parse_residual)
                print('[INFO] seg loss : ', parse_norm.cpu().item())

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def sketch_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2sketch = FaceSketchTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            # print("use class_num")
            class_num = 7
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2sketch.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def landmark_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    img2landmark = FaceLandMarkTool(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = img2landmark.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def arcface_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_path=None):
    idloss = IDLoss(ref_path=ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def arcface_land_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=None, stop=100, id_ref_path=None, land_ref_path=None):
    idloss = IDLoss(ref_path=id_ref_path).cuda()
    img2landmark = FaceLandMarkTool(ref_path=land_ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual) * rho_scale[0]
        # norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        land_residual = img2landmark.get_residual(x0_t)
        land_norm = torch.linalg.norm(land_residual) * rho_scale[1]
        norm_grad = torch.autograd.grad(outputs=norm + land_norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt()
        if not i <= stop:
            xt_next -= rho * norm_grad
        
        x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

        if True or not i <= stop:
            with torch.no_grad():
                residual = idloss.get_residual(xt_next)
                norm = torch.linalg.norm(residual)
                print('[INFO] ID loss : ', norm.cpu().item())
                landmark_residual = img2landmark.get_residual(xt_next)
                landmark_norm = torch.linalg.norm(landmark_residual)
                print('[INFO] landmark loss : ', landmark_norm.cpu().item())

    return [xs[-1]], [x0_preds[-1]]


def arcface_land_ddim_diffusion_magic(x, seq, model, b, cls_fn=None, rho_scale=None, stop=100, id_ref_path=None, land_ref_path=None):
    idloss = IDLoss(ref_path=id_ref_path).cuda()
    img2landmark = FaceLandMarkTool(ref_path=land_ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        
        xt.requires_grad = True
        
        if cls_fn == None:
            et = model(xt, t)
        else:
            print("use class_num")
            class_num = 281
            classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
            et = model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

        if et.size(1) == 6:
            et = et[:, :3]
        
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        residual = idloss.get_residual(x0_t)
        norm = torch.linalg.norm(residual) 
        norm_grad = torch.autograd.grad(outputs=norm, inputs=xt, retain_graph=True)[0]

        # land_residual = img2landmark.get_residual(x0_t)
        # land_norm = torch.linalg.norm(land_residual) * rho_scale[1]
        # norm_grad = torch.autograd.grad(outputs=norm + land_norm, inputs=xt)[0]
        
        eta = 0.5
        c1 = (1 - at_next).sqrt() * eta
        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
        
        # use guided gradient
        rho = at.sqrt() * rho_scale[0]
        if not i <= stop:
            xt_next -= rho * norm_grad
            # calculate x|c1
            xt_land = xt_next.clone()
            et_land = model(xt_land, t)
            
            if et_land.size(1) == 6:
                et_land = et_land[:, :3]
                
            x0_t_land = (xt_land - et_land * (1 - at).sqrt()) / at.sqrt()
            land_residual = img2landmark.get_residual(x0_t_land)
            land_norm = torch.linalg.norm(land_residual)
            land_norm_grad = torch.autograd.grad(outputs=land_norm, inputs=xt_land)[0]
            gradient_prod = torch.dot(xt_next.view(-1), land_norm_grad.detach().view(-1))
            land_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=xt, allow_unused=False)[0]
        
        # use guided gradient
        if not i <= stop:
            land_rho = at.sqrt() * rho_scale[1]
            xt_next -= land_rho * land_loss_magic_grad.detach()
            x0_t = x0_t_land.detach()
        else:
            x0_t = x0_t.detach()
        xt_next = xt_next.detach()
        
        use_mdga = True
        if use_mdga and not i <= stop:
            # mdga_grad = mgd([rho * norm_grad.detach(), parse_rho * parse_loss_magic_grad.detach()])
            ca_grad = cagrad([rho * norm_grad.detach(), land_rho * land_loss_magic_grad.detach()])   
            # xt_next -= mdga_grad + rho * norm_grad.detach() + parse_rho * parse_loss_magic_grad.detach()
            xt_next -= ca_grad + rho * norm_grad.detach() + land_rho * land_loss_magic_grad.detach()
        
        x0_preds.append(x0_t.to('cpu'))
        xs.append(xt_next.to('cpu'))

        if True or not i <= stop:
            with torch.no_grad():
                residual = idloss.get_residual(xt_next)
                norm = torch.linalg.norm(residual)
                print('[INFO] ID loss : ', norm.cpu().item())
                landmark_residual = img2landmark.get_residual(xt_next)
                landmark_norm = torch.linalg.norm(landmark_residual)
                print('[INFO] landmark loss : ', landmark_norm.cpu().item())

    return [xs[-1]], [x0_preds[-1]]


def clip_parse_id_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=None, prompt=None, stop=100, domain="face", ref_path=None, id_ref_path=None):
    clip_encoder = CLIPEncoder().cuda()
    parser = FaceParseTool(ref_path=ref_path).cuda()
    idloss = IDLoss(ref_path=id_ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    # for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        
        for idx in range(repeat):
        
            xt.requires_grad = True
            xt_parse = xt.clone() # .detach()
            
            et = model(xt, t)
            et_parse = model(xt_parse, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # text guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            # if not i <= stop:
            #     print('[INFO] text loss : ', norm.cpu().item())
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            # parse gradient
            if not i <= stop:
                x0_t_parse = (xt_parse - et_parse * (1 - at).sqrt()) / at.sqrt()
                parse_residual = parser.get_residual(x0_t_parse)
                id_residual = idloss.get_residual(x0_t_parse)
                parse_norm = torch.linalg.norm(parse_residual) * rho_scale[1]
                id_norm = torch.linalg.norm(id_residual) * rho_scale[2]
                # print('[INFO] seg loss : ', parse_norm.cpu().item())
                parse_norm_grad = torch.autograd.grad(outputs=parse_norm + id_norm, inputs=xt_parse)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            
            # use guided gradient
            parse_rho = at.sqrt()
            if not i <= stop:
                xt_next -= parse_rho * parse_norm_grad
            
            x0_t = x0_t.detach()
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
        if True or not i <= stop:
            with torch.no_grad():
                residual = clip_encoder.get_residual(xt_next, prompt)
                norm = torch.linalg.norm(residual)
                print('[INFO] text loss : ', norm.cpu().item())
                parse_residual = parser.get_residual(xt_next)
                parse_norm = torch.linalg.norm(parse_residual)
                print('[INFO] seg loss : ', parse_norm.cpu().item())
                residual = idloss.get_residual(xt_next)
                norm = torch.linalg.norm(residual)
                print('[INFO] ID loss : ', norm.cpu().item())

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]


def clip_parse_id_ddim_diffusion_magic(x, seq, model, b, cls_fn=None, rho_scale=None, prompt=None, stop=100, domain="face", ref_path=None, id_ref_path=None):
    clip_encoder = CLIPEncoder().cuda()
    parser = FaceParseTool(ref_path=ref_path).cuda()
    idloss = IDLoss(ref_path=id_ref_path).cuda()

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    # iterate over the timesteps
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if domain == "face":
            repeat = 1
        
        for idx in range(repeat):
            xt.requires_grad = True
            xt_2nd = xt.clone()
            
            et = model(xt, t)
            et_2nd = model(xt_2nd, t)
            
            if et.size(1) == 6:
                et = et[:, :3]
                et_2nd = et_2nd[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_t_2nd = (xt_2nd - et_2nd * (1 - at).sqrt()) / at.sqrt()
            
            # text guided gradient
            residual = clip_encoder.get_residual(x0_t, prompt)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt, retain_graph=True)[0]
            
            residual_2nd = clip_encoder.get_residual(x0_t_2nd, prompt)
            norm_2nd = torch.linalg.norm(residual_2nd)
            norm_grad_2nd = torch.autograd.grad(outputs=norm_2nd, inputs=xt_2nd, retain_graph=True)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            xt_next_2nd = c1 * x0_t_2nd + c2 * xt_2nd + c3 * torch.randn_like(x0_t)
            
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l2 = l1 * 0.02
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()
            
            xt_next -= rho * norm_grad
            xt_next_2nd -= rho * norm_grad_2nd

            # parse gradient
            if not i <= stop:
                # calculate x|c1
                xt_parse = xt_next.clone() # .detach()
                et_parse = model(xt_parse, t)
                
                if et_parse.size(1) == 6:
                    et_parse = et_parse[:, :3]
                
                x0_t_parse = (xt_parse - et_parse * (1 - at).sqrt()) / at.sqrt()
                parse_residual = parser.get_residual(x0_t_parse)
                parse_norm = torch.linalg.norm(parse_residual)
                # print('[INFO] seg loss : ', parse_norm.cpu().item())
                parse_norm_grad = torch.autograd.grad(outputs=parse_norm, inputs=xt_parse)[0]
                gradient_prod = torch.dot(xt_next.view(-1), parse_norm_grad.detach().view(-1))
                parse_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod, inputs=xt, allow_unused=False)[0]
                
                # calculate x|c1
                xt_id = xt_next_2nd.clone() # .detach()
                et_id = model(xt_id, t)
                
                if et_id.size(1) == 6:
                    et_id = et_id[:, :3]
                
                x0_t_id = (xt_id - et_id * (1 - at).sqrt()) / at.sqrt()
                id_residual = idloss.get_residual(x0_t_id)
                id_norm = torch.linalg.norm(id_residual)
                # print('[INFO] seg loss : ', parse_norm.cpu().item())
                id_norm_grad = torch.autograd.grad(outputs=id_norm, inputs=xt_id)[0]
                gradient_prod_2nd = torch.dot(xt_next_2nd.view(-1), id_norm_grad.detach().view(-1))
                id_loss_magic_grad = torch.autograd.grad(outputs=gradient_prod_2nd, inputs=xt_2nd, allow_unused=False)[0]
                
            # use guided gradient
            if not i <= stop:
                parse_rho = at.sqrt() * rho_scale[1]
                xt_next -= parse_rho * parse_loss_magic_grad.detach()
                id_rho = at.sqrt() * rho_scale[2]
                xt_next -= id_rho * id_loss_magic_grad.detach()
                x0_t = x0_t_parse.detach()
            else:
                x0_t = x0_t.detach()
                
            use_mdga = True
            if use_mdga and not i <= stop:
                # mdga_grad = mgd([rho * norm_grad.detach(), parse_rho * parse_loss_magic_grad.detach()])
                ca_grad = cagrad_third_order([rho * norm_grad.detach(), parse_rho * parse_loss_magic_grad.detach(), id_rho * id_loss_magic_grad.detach()])   
                # xt_next -= mdga_grad + rho * norm_grad.detach() + parse_rho * parse_loss_magic_grad.detach()
                xt_next -= ca_grad + rho * norm_grad.detach() + parse_rho * parse_loss_magic_grad.detach() + id_rho * id_loss_magic_grad.detach()
                
            xt_next = xt_next.detach()
            
            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
        
        if True or not i <= stop:
            with torch.no_grad():
                residual = clip_encoder.get_residual(xt_next, prompt)
                norm = torch.linalg.norm(residual)
                print('[INFO] text loss : ', norm.cpu().item())
                parse_residual = parser.get_residual(xt_next)
                parse_norm = torch.linalg.norm(parse_residual)
                print('[INFO] seg loss : ', parse_norm.cpu().item())
                id_residual = idloss.get_residual(xt_next)
                id_norm = torch.linalg.norm(id_residual)
                print('[INFO] ID loss : ', id_norm.cpu().item())

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]