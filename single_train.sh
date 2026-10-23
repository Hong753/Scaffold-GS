scene='/data/hyejun_dataset/HandHeld/Orange_office_Colmap'
exp_name='baseline'
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1
frame_selection_stride=${FRAME_SELECTION_STRIDE:-3}

# example:
bash ./train.sh -d "${scene}" -l "${exp_name}" --gpu "${gpu}" --voxel_size "${voxel_size}" --update_init_factor "${update_init_factor}" --appearance_dim "${appearance_dim}" --ratio "${ratio}" --frame_selection_stride "${frame_selection_stride}" "$@"
