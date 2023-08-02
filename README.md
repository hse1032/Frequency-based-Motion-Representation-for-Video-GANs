# Frequency-based-Motion-Representation-for-Video-GANs

### [IEEE Transaction on Image Processing (TIP)] Pytorch implementation
[[Paper](https://ieeexplore.ieee.org/document/10183834)] [[Project page](https://hse1032.github.io/Frequency-based-Motion-Representation-for-Video-GANs-project-page/)]

<div>
<img src="https://github.com/hse1032/Frequency-based-Motion-Representation-for-Video-GANs/assets/115209649/aaef316b-2c19-47ac-b4c2-57a63b5b72ac" alt="toy full frequency range" style="width:240px; height:240px;"\>
<img src="https://github.com/hse1032/Frequency-based-Motion-Representation-for-Video-GANs/assets/115209649/124e0fbc-644f-46d5-af82-6531e65d9421" alt="toy low frequency range" style="width:240px; height:240px;"\>
<img src="https://github.com/hse1032/Frequency-based-Motion-Representation-for-Video-GANs/assets/115209649/45023d4d-9523-44d6-8c5e-46f21f784feb" alt="toy high frequency range" style="width:240px; height:240px;"\>
</div>



### Abstract
Videos contain motions of various speeds. For example, the motions of one’s head and mouth differ in terms of speed — the head being relatively stable and the mouth moving rapidly as one speaks. Despite its diverse nature, previous video GANs generate video based on a single unified motion representation without considering the aspect of speed. In this paper, we propose a frequency-based motion representation for video GANs to realize the concept of speed in video generation process. In detail, we represent motions as continuous sinusoidal signals of various frequencies by introducing a coordinate-based motion generator. We show, in that case, frequency is highly related to the speed of motion. Based on this observation, we present frequency-aware weight modulation that enables manipulation of motions within a specific range of speed, which could not be achieved with the previous techniques. Extensive experiments validate that the proposed method outperforms state-of-the-art video GANs in terms of generation quality by its capability to model various speed of motions. Furthermore, we also show that our temporally continuous representation enables to further synthesize intermediate and future frames of generated
videos.


## Installation
TODO


## Training
### Dataset structure
For the default setting, we follow the dataset structure of previous video GANs such as StyleGAN-V, which extracts every frame before training.
The dataset should have a directory structure as below:
```
dataset/
    video1/
        - frame1.jpg (or png)
        - frame2.jpg (or png)
        - ...
    video2/
        - frame1.jpg (or png)
        - frame2.jpg (or png)
        - ...
    ...
```
As StyleGAN-V said, we recommend an image quality as 95 when using JPEG format for saving.

### Training video GANs
We provide the training script with default hyperparameters in "run.sh".
What you need to set are the paths of the log and dataset directory.

(Default setting is for training UCF-101 and SkyTimelapse with 256x256 resolution)
```
sh run.sh {log directory} {dataset directory}
```

You may change the below arguments for matching your environment:
```
--ngpus        (the number of GPU you use)
--video_batch  (batch size per GPU for video discriminator)
--image_batch  (batch size per GPU for image discriminator)
--batches      (the number of iterations)
--image_size   (spatial resolution of image)
```
For other arguments, please refer "cfg.py"

Also, we provide the 2D discriminator using time step difference as a condition instead of 3D convnet, which greatly reduces the VRAM and training time.
```
--image_discriminator ImageDiscriminator
--video_discriminator DeltaVideoDiscriminator
```
However, we did not explore a lot with this discriminator architecture, so the performance is not guaranteed.


### Evaluation (FVD)
For evaluation, we borrow the FVD calculation code from StyleGAN-V (https://github.com/universome/stylegan-v).

1. Generate videos from the pre-trained model. You can use "test_codes/generate_videos.py".
```
cd test_codes
python3 generate_videos.py --model {model_path} --output_dir {save_path} --num_videos 2048 --type jpg
```

2. Evaluate the model using the evaluation script.
```
cd metrics (test_codes/metrics)
python3 calc_metrics_for_dataset.py --real_data_path {dataset_path}  --fake_data_path {save_path} \
                                    --mirror 1 --gpus 1 --resolution 256 \
                                    --metrics fvd2048_16f --verbose 0 --use_cache 0
```

### Generating videos for manipulating
We provide the video generation script (test_codes/generate_videos.py)

For a detailed explanation, please refer to the comments in the script.



### Toy dataset
You can synthesize the toy dataset we used in the paper by "toy_dataset/toy_dataset.py" script.

If you run below code snippet, 10000 toy videos are saved in "toy_dataset/sampled_videos/
```
cd toy_dataset
python3 toy_dataset
```


## Acknowledgement
Our code is built upon several open-sourced GANs as below

- MoCoGAN (https://github.com/sergeytulyakov/mocogan)
- CIPS (https://github.com/advimman/CIPS)
- StyleGAN2-pytorch (https://github.com/rosinality/stylegan2-pytorch)
- StyleGAN2 (https://github.com/NVlabs/stylegan2)
- StyleGAN-V (https://github.com/universome/stylegan-v)
- DiffAugment (https://github.com/mit-han-lab/data-efficient-gans)

Thanks for them!


## Bibtex
If you find this repository useful, please use the following entry for citation.
```
@article{hyun2023frequency,
  title={Frequency-based Motion Representation for Video Generative Adversarial Networks},
  author={Hyun, Sangeek and Lew, Jaihyun and Chung, Jiwoo and Kim, Euiyeon and Heo, Jae-Pil},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```
