CUDA_VISIBLE_DEVICES=2 python pose2image.py --seed 1234 --timesteps 100 --prompt "young man, realitic photo" --pose_ref "./test_imgs/pose4.jpg" --id_ref "./test_imgs/id3.png"

CUDA_VISIBLE_DEVICES=2 python scribble2image.py --seed 1234 --timesteps 100 --prompt "bike" --scribble_ref "./test_imgs/s5.png" --style_ref "./test_imgs/xingkong.jpg"
