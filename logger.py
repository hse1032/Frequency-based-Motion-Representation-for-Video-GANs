from torch.utils.tensorboard import SummaryWriter
import numpy as np

from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir, suffix=None):
        if suffix is not None:
            self.writer = SummaryWriter(log_dir, filename_suffix=suffix)
        else:
            self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def histogram_summary(self, tag, var, step):
        self.writer.add_histogram(tag, var, step)

    def image_summary(self, tag, images, step):
        # get numpy array
        images = images.transpose([0, 3, 1, 2])
        for i, img in enumerate(images):
            self.writer.add_image(tag='%s/%d' % (tag, i), img_tensor=img, global_step=step)

    def video_summary(self, tag, videos, step):

        sh = list(videos.shape)
        sh[-1] = 1

        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        img_summaries = []
        for i, vid in enumerate(videos):
            # Concat a video
            v = vid.transpose(1, 2, 3, 0)
            v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
            img = np.concatenate(v, axis=1)[:, :-1, :]

            self.writer.add_image(tag='%s/%d' % (tag, i), img_tensor=img.transpose([2, 0, 1]),
                                  global_step=step)