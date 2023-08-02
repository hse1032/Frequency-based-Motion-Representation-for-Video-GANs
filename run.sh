# What need to change for training
ngpus=8
logdir=$1
data=$2

python3 -m torch.distributed.launch --nproc_per_node=$ngpus --master_port=8888 train.py \
                --data_path ${data} \
                --log_path ${logdir} \
                --image_discriminator ImageDiscriminator \
                --video_discriminator PatchVideoDiscriminator \
                --every_nth 1 \
                --video_batch 2 \
                --image_batch 4 \
                --freq_val 7 \
                --ngpus $ngpus \
                --DiffAugment \
                --mixing_prob 0.0 \
                --dim_z_content 512 \
                --dim_z_motion 512 \
                --VD_resize_ratio 0.5 \
                --image_size 256 \
                --batches 300000
