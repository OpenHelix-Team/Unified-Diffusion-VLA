# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
policy_model=openvla
module load ffmpeg/6.0.1
scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028
export DISPLAY='' 
unset DISPLAY
export SVULKAN2_HEADLESS=1
export SAPIEN_NO_WINDOW=1

# export CUDA_VISIBLE_DEVICES='0'
export CUDA_VISIBLE_DEVICES=0
# VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export PYTHONFAULTHANDLER=1
timestamp=$(date +%Y%m%d%H%M)
# ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_SIMPLERENV_BRIDGE_VIDEO_BS64_40k_s128/checkpoint-44000"
ckpt_dir="/data/user/wsong890/user68/project/UniVLA/logs/UNIVLA_SIMPLERENV_BRIDGE_VIDEO_BS256_60k_a10"
# fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_t5_s50"
fast_path="/data/user/wsong890/user68/project/UniVLA/pretrain/fast_bridge_trainval_t10_s50"
steps=128
predict_action_frames=10
max_new_action_len=72
max_new_tokens=708
python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --dis_i2a \
  --steps $steps \
  --log_name ./log/result_udvla_${timestamp} \
  --fast_path ${fast_path} \
  --predict_action_frames $predict_action_frames \
  --max_new_tokens $max_new_tokens \
  --new_action_len $max_new_action_len

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --dis_i2a \
  --steps $steps \
  --log_name ./log/result_udvla_${timestamp} \
  --fast_path ${fast_path} \
  --predict_action_frames $predict_action_frames \
  --max_new_tokens $max_new_tokens \
  --new_action_len $max_new_action_len

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --dis_i2a \
  --steps $steps \
  --log_name ./log/result_udvla_${timestamp} \
  --fast_path ${fast_path} \
  --predict_action_frames $predict_action_frames \
  --max_new_tokens $max_new_tokens \
  --new_action_len $max_new_action_len

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --dis_i2a \
  --steps $steps \
  --log_name ./log/result_udvla_${timestamp} \
  --fast_path ${fast_path} \
  --predict_action_frames $predict_action_frames \
  --max_new_tokens $max_new_tokens \
  --new_action_len $max_new_action_len