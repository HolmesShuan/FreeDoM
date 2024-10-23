# Baseline : ControlNet Scribble Condition + Style Condition (CLIP)
# CUDA_VISIBLE_DEVICES=0 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/1.png" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --skip_grid --precision full --use_clip_style_loss > scribble_clip.txt

# Magic : ControlNet Scribble Condition + Style Condition (CLIP)
CUDA_VISIBLE_DEVICES=3 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/1.png" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --skip_grid --use_magic --style_start_steps 100 --style_end_steps 30 --repeat 3 --time_reverse_step --use_clip_style_loss --style_rho_scale 0.2 --precision full > scribble_magic_clip.txt

# Baseline : ControlNet Scribble Condition + Style Condition (VGG)
CUDA_VISIBLE_DEVICES=3 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/1.png" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --skip_grid --precision full > scribble_vgg.txt

# Magic : ControlNet Scribble Condition + Style Condition (VGG)
CUDA_VISIBLE_DEVICES=3 python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/1.png" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --skip_grid --use_magic --style_start_steps 100 --style_end_steps 30 --repeat 3 --time_reverse_step --style_rho_scale 0.2 --precision full > scribble_magic_vgg.txt

# CUDA_VISIBLE_DEVICES=0 python txt2img.py --prompt "sky." --style_ref_img_path "./style_images/1.png" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --skip_grid
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/bijiasuo.jpeg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/bw.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/jojo.jpeg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/nahan.jpeg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/tan.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/xiangrikui.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/xing.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0
# python txt2img.py --prompt "a cat wearing glasses." --style_ref_img_path "./style_images/xingkong.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0