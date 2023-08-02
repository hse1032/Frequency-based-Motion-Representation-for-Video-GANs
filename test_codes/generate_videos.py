import os
import torch

import numpy as np
import cv2
import imageio
import skvideo.io

def save_video_custom(video, filename, type):
    assert type in ['gif', 'mp4', 'png', 'jpg', 'png_cat', 'jpg_cat']
    save_type = args.type if args.type not in ['jpg_cat', 'png_cat'] else args.type.split('_')[0]
    # save as GIF
    if type == 'gif':
        with imageio.get_writer(filename + '.{}'.format(save_type), mode='I') as writer:
            for v in video:
                writer.append_data(v)
    # save as img (concatenated)
    elif type in ['png_cat', 'jpg_cat']:
        T, H, W, C = video.shape
        vid_saved = np.zeros([H, W*T, C], dtype=np.uint8)
        for idx, v in enumerate(video):
            vid_saved[:, W*idx:W*(idx+1), :] = v
        cv2.imwrite(filename + '.{}'.format(save_type), cv2.cvtColor(vid_saved, cv2.COLOR_RGB2BGR))
    elif type in ['png', 'jpg']:
        os.makedirs(filename, exist_ok=True)
        for idx, v in enumerate(video):
            if type == 'png':
                cv2.imwrite(os.path.join(filename, '{:03d}.{}'.format(idx, save_type)), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
            elif type == 'jpg':
                cv2.imwrite(os.path.join(filename, '{:03d}.{}'.format(idx, save_type)), cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                
    # save as mp4
    else:
        skvideo.io.vwrite(os.path.join(filename + '.{}'.format(save_type)), video, outputdict={"-vcodec": "libx264"})

def repeating_frame(video, ratio):
    return np.repeat(video, ratio, axis=0)

import sys
sys.path.append('..')

from models import generator as generator_models
from trainers import videos_to_numpy
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help="path for pre-trained model")
    parser.add_argument('--num_videos', type=int, help="number of videos to generate")
    parser.add_argument('--output_dir', type=str, help="output directory")
    parser.add_argument('--type', type=str, help="save type of file [mp4, gif, png, jpg]")

    args = parser.parse_args()

    # Load pretrained model
    ckpt = torch.load(args.model, map_location={'cuda:0': 'cpu'})
    args_model = ckpt['args']
    
    
    freqs = []
    for f in args_model.freq_val.split(','):
        freqs.append(int(f))
        

    generator = generator_models.VideoGenerator(args_model.image_size, args_model.dim_z_content,
                                      args_model.dim_z_motion, args_model.video_length, freqs=freqs)

    generator.load_state_dict(ckpt['g_ema'])
    generator.eval().cuda()

    num_videos = int(args.num_videos)
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # =============== how to generate interpolated & extrapolated videos ==================
    # time_steps --> number of frames to generator
    # grid_ranges --> range of sampling coordinate
    # Default time_steps = 16
    # Default grid_ranges = (-1, 1)
    # e.g. if "time_step = [16, 32]" and "grid_ranges = [(-1, 1), (-1, 3)]
    #      it generates both basic video and 2 times longer video
    time_steps = [16]
    grid_ranges = [(-1, 1)]

    # =============== how to define range of frequency to manipulate ==================
    # freqs --> ranges to manipulate
    # e.g. [3., 5., 1e+8] means we divides where to apply motion code into 3 ranges
    #      (0, 3.), (3., 5.), (5., 1e+8)

    freqs = [1e+8]

    # you can use below variables to get specific range of frequency
    # Maximum value of learned frequency ([ ] float number)
    # --> max_freq = np.max(np.abs(generator.LFF[0].ffm.linear.weight.data.numpy()[:, 0]))
    # Sorted array of frequencies ([512] vector)
    # --> sorted_freqs = np.sort(np.abs(generator.LFF[0].ffm.linear.weight.data.numpy()[:, 0]))

    with torch.no_grad():
        # generate 1 video for a loop
        for i in range(num_videos):
            # basic format (single motion code)
            z_m = generator.sample_z_m(1) # motion code
            z_c = generator.sample_z_content(1) # content code
            zs = [z_m, z_c]
            nonzeros = None # range index to leave (None for leaving all ranges)


            # =============== how to manipulate motion and content ==================
            # you can fix z_c or z_m by using previously sampled code
            # ---------------------------------------------------------
            # z_m1 = generator.sample_z_m(1)
            # z_m2 = generator.sample_z_m(1)
            # z_c = generator.sample_z_content(1)
            # zs_1 = [z_m1, z_c]
            # zs_2 = [z_m2, z_c]  --> z_s1, z_s2 generate same content different motion
            # nonzeros = None
            # ---------------------------------------------------------


            # =============== how to remove specific range of frequency ==================
            # if you want to remove motion of specific range of frequencies, do like this
            # ---------------------------------------------------------
            # freqs = [k, 1e+8] --> divide range (0, k), (k, 1e+8)
            # z_m = generator.sample_z_m(1)
            # z_c = generator.sample_z_content(1)
            # zs = [z_m, z_c]
            # nonzeros = [0] --> only leave range index 0, (0, k)
            # ---------------------------------------------------------
            # In that case, generated videos has only range of frequencies (0, k)


            # =============== how to manipulate motion in specific frequency ==================
            # if you want to use multiple motion code for specific range of frequencies, do like this
            # ---------------------------------------------------------
            # freqs = [k, 1e+8] --> divide range (0, k), (k, 1e+8)
            # z_m1 = generator.sample_z_m(1)
            # z_m2 = generator.sample_z_m(1)
            # z_m = [z_m1, z_m2] --> z_m1 is applied to (0, k) / z_m2 is applied to (k, 1e+8)
            # z_c = generator.sample_z_content(1)
            # zs = [z_m, z_c]
            # nonzeros = None
            # ---------------------------------------------------------
            # In that case, z_m1 is applied to frequency of range (freqs[idx-1], freqs[idx])
            # e.g.
            # freqs = [3.] --> (0, 3.) for z_m1, (3., -) for z_m2
            # freqs = [3., 5.] --> (0, 3.) for z_m1, (3., 5.) for z_m2, (5., -) for z_m3


            video_noise = generator.sample_video_noise(1, 16)
            for t, r in zip(time_steps, grid_ranges):
                v, fidx = generator.sample_videos_from_zs_freqs_zeros(1, zs, t, freqs=freqs, nonzeros=nonzeros, video_noise=video_noise, grid_range=r)
                video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
                # maybe you should change the output file name
                save_video_custom(video, os.path.join(output_dir, "{:03d}_t{}_r{}".format(i, t, r)), type=args.type)
