# Magic : Inversed Background + Cropped Content Condition (VGG ReLU2_2) + Stylized Content Condition (VGG)
# CUDA_VISIBLE_DEVICES=1,2 python txt2img.py --prompt "cat" --background_ref_img_path "./bg/bg52.png" --style_ref_img_path "./bg/bg52.png"  --content_ref_img_path "./content/cat.png" --content_ref_mask_path "./content/cat_mask.npy" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --content_start_steps 40 --content_end_steps 0 --style_start_steps 40 --style_end_steps 0 --repeat 3 --time_reverse_step --precision full --use_cagrad

# Magic : Inversed Background + Cropped Content Condition (VGG ReLU2_2) + Stylized Content Condition (CLIP)
# CUDA_VISIBLE_DEVICES=1,2 python txt2img.py --prompt "cat" --background_ref_img_path "./bg/bg52.png" --style_ref_img_path "./bg/bg52.png"  --content_ref_img_path "./content/cat.png" --content_ref_mask_path "./content/cat_mask.npy" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --content_start_steps 40 --content_end_steps 0 --style_start_steps 40 --style_end_steps 0 --repeat 3 --time_reverse_step --precision full --use_cagrad --use_clip_style_loss

# Magic : Content Condition (VGG ReLU2_2) + Style Condition (VGG)
# CUDA_VISIBLE_DEVICES=1 python txt2img.py --prompt "*" --style_ref_img_path "./bg/bg52.png" --content_ref_img_path "./content/catjpg.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --content_start_steps 100 --content_end_steps 0 --style_start_steps 40 --style_end_steps 0 --repeat 3 --time_reverse_step --precision full --use_cagrad --style_rho_scale 10.0

# Magic : Content Condition (VGG ReLU2_2) + Style Condition (CLIP)
CUDA_VISIBLE_DEVICES=1 python txt2img.py --prompt "*" --style_ref_img_path "./style_images/bijiasuo.jpeg" --content_ref_img_path "./content/catjpg.jpg" --ddim_steps 100 --n_iter 1 --n_samples 1 --seed 2023 --H 512 --W 512 --scale 5.0 --content_start_steps 100 --content_end_steps 0 --style_start_steps 100 --style_end_steps 0 --repeat 3 --time_reverse_step --use_cagrad --use_clip_style_loss --style_rho_scale 10.0 --precision full