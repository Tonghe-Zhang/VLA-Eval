## Stateless policy pi05 on 4 simpler tasks on 1 gpu
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
# debug aid
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
# framework: disable tf and use torch instead
export USE_TF=0
export USE_TORCH=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="/home/tonghez/Project/lerobot:$PYTHONPATH"
export PYTHONPATH="/home/tonghez/Project:$PYTHONPATH"
export PYTHONPATH="/home/tonghez/Project/openpi-main/src:$PYTHONPATH"
export PYTHONPATH="/home/tonghez/Project/openpi-main/packages/openpi-client/src:$PYTHONPATH"
cd /home/tonghe/Project/Manip/quick_test_maniskill/evaluate
source /home/tonghe/Project/openpi-main-portable/.venv/bin/activate
TASK=StackGreenCubeOnYellowCubeBakedTexInScene-v1 #PutSpoonOnTableClothInScene-v1 #PutCarrotOnPlateInScene-v1 #StackGreenCubeOnYellowCubeBakedTexInScene-v1 # PutSpoonOnTableClothInScene-v1  # PutCarrotOnPlateInScene-v1 #  PutEggplantInBasketScene
python3 load_openpi.py \
--config_name pi05_maniskill \
--tasks maniskill_${TASK} \
--mode maniskill \
--exp_name pi05_stateless_${TASK} \
--action_horizon 5 \
--action_chunk 5 \
--num_steps 4 \
--model_device cuda:0
