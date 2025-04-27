#!/bin/bash

python RobustnessEvaluator.py --run_name rotation --r_degree 45 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_45
python RobustnessEvaluator.py --run_name rotation --r_degree 90 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_90
python RobustnessEvaluator.py --run_name rotation --r_degree 135 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_135
python RobustnessEvaluator.py --run_name rotation --r_degree 180 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_180
python RobustnessEvaluator.py --run_name rotation --r_degree 225 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_225
python RobustnessEvaluator.py --run_name rotation --r_degree 270 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_270
python RobustnessEvaluator.py --run_name rotation --r_degree 315 --num_samples 1000 --output_dir /baai-cwm-nas/algorithm/ziyang.yan/nips_2025/vae_data/eval/Gustavosta/rotation_315