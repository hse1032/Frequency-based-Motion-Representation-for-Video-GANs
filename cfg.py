import argparse


def parse_args():

    parser = argparse.ArgumentParser('Configurations')
    parser.add_argument('--random_seed', type=int, default='12345')
    
    # Dataset
    parser.add_argument('--data_path', type=str, help='path to data directory')
    
    parser.add_argument('--image_dataset', type=str, help='specifies a separate dataset to train for images [default: ]')
    parser.add_argument('--image_batch', type=int, default=16, help='number of iamges in image batch PER GPU [default: 16]')
    parser.add_argument('--video_batch', type=int, default=8, help='number of videos in video batch PER GPU [default: 8]')
    parser.add_argument('--image_size', type=int, default=64, help='resize all frames to this size')
    parser.add_argument('--video_length', type=int, default=16, help='length of the video')
    parser.add_argument('--every_nth', type=int, default=2, help='sample training videos using every nth frame (frame rate)')

    # Discriminator architecture
    parser.add_argument('--video_discriminator', type=str, required=True, help='specifies video disciminator type (see models.py for a list of available models)')
    parser.add_argument('--image_discriminator', type=str, required=True, help='specifies image disciminator type (see models.py for a list of available models)')
    parser.add_argument("--VD_resize_ratio", type=float, default=1.0, help='resize videos when giving it to video dicsriminator')

    # Generator options
    parser.add_argument('--dim_z_content', type=int, default=512, help='dimensionality of the content input, i.e. hidden space')
    parser.add_argument('--dim_z_motion', type=int, default=512, help='dimensionality of the motion input')
    parser.add_argument('--freq_val', type=str, default='7', help="seperated by comma e.g. 3,5,7 to [3, 5, 7]")

    # training options
    parser.add_argument('--batches', type=int, default=500000, help='specify number of batches to train (number of total iterations)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load ckpt from path (for resuming training)')

    # Diffaug options
    parser.add_argument("--DiffAugment", action="store_true", help='use diffaug (color,translation,cutout,flip)')
    
    # Regularizations
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--mixing_prob", type=float, default=0.0, help='probability for mixing motion code (Not used)')
    parser.add_argument("--ema_kimg", type=float, default=10, help='halflife of ema weight')

    # Logs
    parser.add_argument('--log_path', type=str, help='path to log directory')
    parser.add_argument('--save_interval', type=int, default=20000, help='interaval for saving ckpts')
    parser.add_argument('--G_save_interval', type=int, default=5000, help='interval for saving generator ckpts (for evaluating FVDs later)')
    parser.add_argument('--print_every', type=int, default=100, help='print every iterations')
    parser.add_argument('--num_print_images', type=int, default=4, help='number of images (videos) for tensorboard')

    # Multi GPU options
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--distributed", action="store_true")
    
    # When training ucf from mp4 file directly, do center crop
    parser.add_argument("--center_crop", action="store_true")
    
    args = parser.parse_args()

    return args
