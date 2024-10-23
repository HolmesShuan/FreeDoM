# Baseline : ControlNet Pose Condition + ID 1 Condition
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "a young man, realistic photo" --pose_ref "./test_imgs/pose4.jpg" --id_ref "./test_imgs/id3.png" --output_prefix "baseline_portrait_photo" > portrait.txt

# Magic : ControlNet Pose Condition + ID 1 Condition
CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "a young man, realistic photo" --pose_ref "./test_imgs/pose4.jpg" --id_ref "./test_imgs/id3.png" --use_magic --output_prefix "magic_portrait_photo" --rho_scale 0.08 --start_steps 100 --end_steps 0 --time_reverse_step > portrait_magic.txt

# Baseline : ControlNet Pose Condition + ID 1 Condition + ID 2 Condition
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "two young men, realistic photo" --pose_ref "./test_imgs/couple_v2.jpeg" --id_ref "./test_imgs/id3.png" --id_ref_2nd "./test_imgs/id7.jpg" --output_prefix "baseline_group_photo" > couple.txt

# Magic : ControlNet Pose Condition + ID 1 Condition + ID 2 Condition
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "two young men, realistic photo" --pose_ref "./test_imgs/couple_v2.jpeg" --id_ref "./test_imgs/id3.png" --id_ref_2nd "./test_imgs/id7.jpg" --use_cagrad --cagrad_weight 0.08 --use_magic --output_prefix "magic_group_photo" > couple_magic.txt



# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 37837 --timesteps 100 --prompt "young man, realitic photo" --pose_ref "./test_imgs/pose4.jpg" --id_ref "./test_imgs/id3.png"
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "two young men, realistic photo" --pose_ref "./test_imgs/couple.jpeg" --id_ref "./test_imgs/id3.png"
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "two young men, realistic photo" --pose_ref "./test_imgs/couple_v2.jpeg" --id_ref "./test_imgs/id3.png"
# CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 3737 --timesteps 100 --prompt "two young men, realistic photo" --pose_ref "./test_imgs/couple.jpg" --id_ref "./test_imgs/id3.png"

# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 95 --start_steps 95 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 95 --start_steps 95 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 90 --start_steps 90 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 85 --start_steps 85 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 80 --start_steps 80 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 75 --start_steps 75 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 70 --start_steps 70 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 512

# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 100 --start_steps 100 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 95 --start_steps 95 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 90 --start_steps 90 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 85 --start_steps 85 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 80 --start_steps 80 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 75 --start_steps 75 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 70 --start_steps 70 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 224

# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 100 --start_steps 100 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 95 --start_steps 95 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 90 --start_steps 90 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 85 --start_steps 85 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 80 --start_steps 80 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 75 --start_steps 75 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
# CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 70 --start_steps 70 --end_steps 0 --lr 0.7 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg" --image_size 384
