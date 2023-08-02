import cv2
import numpy as np

class toy_dataset():

    def __init__(self, image_size, ball_size, base_speed=[1,2,3], speed_multiplier=3):
        self.image_size = image_size
        self.ball_size = ball_size

        self.speed_multiplier = speed_multiplier
        self.base_speed = np.array(base_speed) * speed_multiplier

        self.coords = None
        self.directions = None

        self.shapes = ['circle', 'square', 'triangle']

    def init_coords(self):
        coords = []
        directions = []
        for i in range(3):
            coords.append([int(self.image_size * (i + 1) / 4), 
                np.random.randint(self.ball_size, self.image_size - self.ball_size)])
            directions.append(np.random.randint(2) * 2 - 1)

        return coords, directions

    def generate_frame(self, coords, shapes, colors):
        frame = np.zeros([self.image_size, self.image_size, 3])
        
        for coord, shape, color in zip(coords, shapes, colors):
            if shape == 'circle':
                cv2.circle(frame, coord, self.ball_size, color=color, thickness=-1)
            elif shape == 'square':
                coord = np.array(coord)
                cv2.rectangle(frame, coord - self.ball_size + 1, coord + self.ball_size - 1, color=color, thickness=-1)
            elif shape == 'triangle':
                draw_triangle(frame, coord, self.ball_size * 3, color=color)
            else:
                assert shape in self.shapes
                exit(-1)

        return frame

    def next_frame(self, coords, directions, speeds):
        new_coords = []
        for idx, (delta, coord) in enumerate(zip(speeds, coords)):
            delta = delta * directions[idx]

            if coord[1] + delta > self.image_size - self.ball_size:
                directions[idx] *= -1
                delta = - (coord[1] + delta - self.image_size + self.ball_size)
            elif coord[1] + delta < self.ball_size:
                directions[idx] *= -1
                delta = - (coord[1] + delta - self.ball_size)

            new_coords.append([coord[0], coord[1] + delta])
        return new_coords, directions
    
    def generate_video(self, video_len, coords, directions):
        frames = []

        shapes, colors, speeds = self.get_params()
        for i in range(video_len):
            frames.append(self.generate_frame(coords, shapes, colors))
            coords, directions = self.next_frame(coords, directions, speeds)

        return frames

    def get_params(self):
        shapes = np.random.choice(self.shapes, 3)
        colors = [get_random_color(), get_random_color(), get_random_color()]
        speeds = [np.random.choice(self.base_speed), np.random.choice(self.base_speed), np.random.choice(self.base_speed)]
        return shapes, colors, speeds

def draw_triangle(frame, center, size, color):
    center_np = np.array(center)
    xlen, ylen = np.array([int(size / 3), 0]), np.array([0, int(size / 3)])
    pts = [center_np - xlen + ylen, center_np - ylen, center_np + ylen + xlen]
    cv2.fillPoly(frame, np.array([pts]), color)

def get_random_color():
    return [int(i) for i in np.random.randint(100, 255, size=(3), dtype=np.uint8)]

# Saving examples from toy dataset 
if __name__ == "__main__":
    import imageio
    import skvideo.io
    import os
    from PIL import Image

    def save_video_mp4(video, filename):
        skvideo.io.vwrite(os.path.join(filename), video, outputdict={"-vcodec": "libx264"})

    def save_video_gif(video, filename):
        frames = []
        for f in video:
            frames.append(Image.fromarray(f.astype(np.uint8)))
        imageio.mimsave(filename, frames)

    def save_video_img(video, filename):
        t, h, w, c = np.shape(video)
        
        image = np.zeros([h, w*t, c])
        for idx, f in enumerate(video):
            image[:, idx*w:(idx+1)*w] = f
        cv2.imwrite(filename, image)

    def save_video_img_divided(video, filename):
        os.makedirs(filename, exist_ok=True)

        for idx, f in enumerate(video):
            cv2.imwrite(os.path.join(filename, "{:02d}.png".format(idx)), f)


    data_gen = toy_dataset(64, 4, [1, 2, 4], speed_multiplier=2)

    coords, directions = data_gen.init_coords()
    frames = data_gen.generate_video(100, coords, directions)
    
    os.makedirs("sampled_videos/", exist_ok=True)
    for i in range(10000):
        coords, directions = data_gen.init_coords()
        frames = data_gen.generate_video(32, coords, directions)
        
        save_video_img_divided(frames, "sampled_videos/")

