import os


import numpy as np
import torch.utils.data

import glob
import random
from PIL import Image
import av


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, video_length, every_nth=1, img_transform=None, vid_transform=None, delta_t=False):
        self.data_path = data_path
        self.step = every_nth
        self.video_length = video_length
        
        self.delta_t = delta_t # delta_t for DeltaVideoDiscriminator

        vids = glob.glob(self.data_path + '/*')

        self.vids = []
        for v in vids:
            if len(glob.glob(v + '/*')) < self.video_length * self.step:
                print("Pass video:", v)
            else:
                self.vids.append(v)

        self.img_transform = img_transform
        self.vid_transform = vid_transform

    def __getitem__(self, idx):
        
        video_path = self.vids[idx]
        frames = sorted(glob.glob(video_path + '/*'))
        nframes = len(frames)

        start_idx = random.randint(0, nframes - self.video_length * self.step)
        
        if self.delta_t:
            frame_idx = sorted(np.random.choice(range(self.video_length), 2, replace=False) * self.step)
            vid = [Image.open(frames[frame_idx[0]]).convert('RGB'), Image.open(frames[frame_idx[1]]).convert('RGB')]
            delta = torch.FloatTensor([((frame_idx[1] - frame_idx[0]) / (self.video_length * self.step)) * 2.]) # make delta range in (-1, 1)
        else:    
            vid = [Image.open(frames[start_idx + i * self.step]).convert('RGB') for i in range(self.video_length)]
            delta = torch.FloatTensor([0]) # dummy tensor

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)

        frame_idx = random.randint(0, nframes - 1)
        img = Image.open(frames[frame_idx]).convert('RGB')

        if self.img_transform is not None:
            img = self.img_transform(img)

        # Do not use categorical labels, currently
        return {"images": img, "videos": vid, "categories": torch.LongTensor([0]), "delta": delta}

    def __len__(self):
        return len(self.vids)

class Dataset_mp4(torch.utils.data.Dataset):
    def __init__(self, data_path, video_length, every_nth=1, img_transform=None, vid_transform=None):
        self.data_path = data_path
        self.step = every_nth
        self.video_length = video_length

        self.dirs = glob.glob(self.data_path + '/*')
        self.vids = []
        for dname in self.dirs:
            if os.path.isdir(dname):
                self.vids += glob.glob(dname + '/*')

        self.img_transform = img_transform
        self.vid_transform = vid_transform

    def __getitem__(self, idx):
        video_path = self.vids[idx]
        
        frames = []
        container = av.open(video_path)
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
        nframes = len(frames)
        if nframes < self.video_length * self.step:
            print(video_path)
        start_idx = random.randint(0, nframes - self.video_length * self.step)
        vid = [frames[start_idx + i * self.step] for i in range(self.video_length)]

        if self.vid_transform is not None:
            vid = self.vid_transform(vid)

        frame_idx = random.randint(0, nframes - 1)
        img = frames[frame_idx]

        if self.img_transform is not None:
            img = self.img_transform(img)

        # Do not use categorical labels, currently
        return {"images": img, "videos": vid, "categories": torch.LongTensor([0])}

    def __len__(self):
        return len(self.vids)

from toy_dataset.toy_dataset import toy_dataset

class Dataset_toy(torch.utils.data.Dataset):
    def __init__(self, video_length, every_nth=1, img_transform=None, vid_transform=None):

        self.step = every_nth
        self.video_length = video_length

        self.img_transform = img_transform
        self.vid_transform = vid_transform

        self.video_generator = toy_dataset(64, 4, base_speed=[2 ** i for i in range(3)], speed_multiplier=2)

    def __getitem__(self, idx):
        coords, directions = self.video_generator.init_coords()
        frames = self.video_generator.generate_video(16, coords, directions)
        vid = [Image.fromarray(frames[i].astype(np.uint8)) for i in range(len(frames))]
        img = vid[np.random.randint(len(frames))]
        if self.vid_transform is not None:
            vid = self.vid_transform(vid)
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Do not use categorical labels, currently
        return {"images": img, "videos": vid, "categories": torch.LongTensor([0])}

    def __len__(self):
        return int(1e+8)
