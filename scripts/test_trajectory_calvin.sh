main_dir=Planner_Calvin

dataset=./data/calvin/packaged_D_D/training
valset=./data/calvin/packaged_D_D/validation

lr=3e-4
wd=5e-3
dense_interpolation=1
interpolation_length=20
num_history=3
diffusion_timesteps=25
B=30
C=168
ngpus=5
backbone=clip
image_size="256,256"
relative_action=1
fps_subsampling_factor=3
lang_enhanced=1
gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
gripper_buffer=0.01
val_freq=5000
quaternion_format=wxyz
train_iters=100000
use_pcd=1

model_name=2024-07-20T04:30:31+00:00-use_pcd:1

log_dir=train_logs/standalone_eval/${main_dir}/${model_name}/pcd_1_eval/

export PYTHONPATH=`pwd`:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

mkdir -p $log_dir
cp $0 $log_dir
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    online_evaluation_calvin/evaluate_policy.py \
    --calvin_dataset_path calvin/dataset/task_D_D \
    --calvin_model_path calvin/calvin_models \
    --text_encoder clip \
    --text_max_length 16 \
    --tasks D\
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
    --save_video 0 \
    --base_log_dir $log_dir \
    --quaternion_format $quaternion_format \
    --checkpoint train_logs/${main_dir}/${model_name}/last.pth \
    --use_pcd $use_pcd