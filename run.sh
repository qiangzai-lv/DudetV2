bash tools/dist_train.sh projects/Dudet/config/vggtdet_scannet.py 1



bash tools/dist_test.sh projects/Dudet/config/vggtdet_scannet.py /mnt/workspace/code/dudet/mmdetection3d/work_dirs/detr3d_scannet_rgb_only/epoch_4.pth 1
