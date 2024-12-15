#!/bin/bash

#SBATCH -p accelerated
#SBATCH -J test_3dda_omer

#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 05:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

source /home/hk-project-sustainebot/uqtlv/.bashrc
conda activate 3d_diffuser_actor_dev

main_dir=Planner_Calvin

dataset=./data/calvin/packaged_D_D/training
valset=./data/calvin/packaged_D_D/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=1
diffusion_timesteps=25
B=30
C=192
ngpus=4
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
input_mode=3d

export PYTHONPATH=`pwd`:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# pat=/home/hk-project-sustainebot/uqtlv/code/3d_diffuser_actor/train_logs/omer/2d3dmix-2024-10-18T16:43:06+02:00
pat=/hkfs/work/workspace/scratch/uqtlv-code_n_mamba/code/3d_diffuser_actor/train_logs/omer/3donly-2024-10-19T18:21:49+02:00
checkpoint=0004999

torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path calvin/dataset/task_D_D \
    --calvin_model_path calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks A B C D\
    --backbone $backbone \
    --gripper_loc_bounds $gripper_loc_bounds \
    --gripper_loc_bounds_buffer $gripper_buffer \
    --calvin_gripper_loc_bounds calvin/dataset/task_D_D/validation/statistics.yaml \
    --embedding_dim $C \
    --action_dim 7 \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --interpolation_length $interpolation_length \
    --num_history $num_history \
    --relative_action $relative_action \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --save_video 1 \
    --base_log_dir $pat/eval_logs/$input_mode-test1/ \
    --quaternion_format $quaternion_format \
    --checkpoint $pat/$checkpoint.pth \
    --input_mode $input_mode
