export CUDA_VISIBLE_DEVICES=0
python test.py --checkpoint_path logs/baseline/checkpoint.tar --camera realsense --dataset_root ./grasp_data --batch_size 3
