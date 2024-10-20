# Baseline : Face Parse Condition + Text Control
CUDA_VISIBLE_DEVICES=3 python main.py -i clip_parse_face -s clip_parse_ddim --doc celeba_hq --timesteps 100 --seg_rho_scale 0.2 --seed 12345 --stop 200 --model_type "face" --prompt "blonde beauty" --ref_path ./images/294.jpg --batch_size 1 > clip_parse.txt

# Magic : Face Parse Condition + Text Control
CUDA_VISIBLE_DEVICES=3 python main.py -i clip_parse_face_magic -s clip_parse_ddim_magic --doc celeba_hq --timesteps 100 --seg_rho_scale 0.1 --seed 12345 --stop 200 --model_type "face" --prompt "blonde beauty" --ref_path ./images/294.jpg --batch_size 1 > clip_parse_magic.txt

# Baseline : Face Landmark Condition + Face ID
CUDA_VISIBLE_DEVICES=3 python main.py -i arc_land_ddim -s arc_land_ddim --doc celeba_hq --timesteps 100 --land_rho_scale 500 --id_rho_scale 100 --seed 12345 --stop 200 --model_type "face" --prompt "blonde beauty" --id_ref_path ./images/id10.png --land_ref_path ./images/294.jpg --batch_size 1 > id_land.txt

# Magic : Face Landmark Condition + Face ID
CUDA_VISIBLE_DEVICES=3 python main.py -i arc_land_ddim_magic -s arc_land_ddim_magic --doc celeba_hq --timesteps 100 --land_rho_scale 500 --id_rho_scale 100 --seed 1234 --stop 200 --model_type "face" --prompt "blonde beauty" --id_ref_path ./images/id10.png --land_ref_path ./images/294.jpg --batch_size 1 > id_land_magic.txt

# Baseline : Face Parse Condition + Face ID + Text Control
CUDA_VISIBLE_DEVICES=3 python main.py -i clip_parse_id_ddim -s clip_parse_id_ddim --doc celeba_hq --timesteps 100 --land_rho_scale 500 --id_rho_scale 100 --seed 12345 --stop 200 --model_type "face" --prompt "blonde beauty" --ref_path ./images/294.jpg --id_ref_path ./images/id10.png --batch_size 1 > clip_parse_id.txt

# Magic : Face Parse Condition + Face ID + Text Control
CUDA_VISIBLE_DEVICES=3 python main.py -i clip_parse_id_ddim_magic -s clip_parse_id_ddim_magic --doc celeba_hq --timesteps 100 --land_rho_scale 500 --id_rho_scale 100 --seed 12345 --stop 200 --model_type "face" --prompt "blonde beauty" --ref_path ./images/294.jpg --id_ref_path ./images/id10.png --batch_size 1 > clip_parse_id_magic.txt

# CUDA_VISIBLE_DEVICES=1 python main.py -i clip_face -s clip_ddim --doc celeba_hq --timesteps 100 --seed 12345 --model_type "face" --prompt "blonde beauty" --batch_size 1
# CUDA_VISIBLE_DEVICES=1 python main.py -i seg_face -s parse_ddim --doc celeba_hq --timesteps 100 --rho_scale 0.2 --seed 12345 --stop 200 --ref_path ./images/294.jpg --batch_size 1
# CUDA_VISIBLE_DEVICES=3 python main.py -i sketch_face -s sketch_ddim --doc celeba_hq --timesteps 100 --rho_scale 20.0 --seed 1234 --stop 100 --ref_path ./images/294.jpg --batch_size 1
# CUDA_VISIBLE_DEVICES=3 python main.py -i landmark_face -s land_ddim --doc celeba_hq --timesteps 100 --rho_scale 500.0 --seed 1234 --stop 200 --ref_path ./images/2334.jpg --batch_size 1
# CUDA_VISIBLE_DEVICES=3 python main.py -i arcface_face -s arc_ddim --doc celeba_hq --timesteps 100 --rho_scale 100.0 --seed 1234 --stop 100 --ref_path  ./images/id10.png --batch_size 1



# # imagenet
# CUDA_VISIBLE_DEVICES=3 python main.py -i clip_imagenet -s clip_ddim --doc imagenet --timesteps 100 --seed 1234 --model_type "imagenet" --prompt "orange" --batch_size 1
