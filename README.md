<div align="center">
<h1>HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting</h1>

[**Zhenglin Zhou**](https://scholar.google.com/citations?user=6v7tOfEAAAAJ) · [**Fan Ma**](https://flowerfan.site/) · [**Hehe Fan**](https://hehefan.github.io/) · [**Zongxin Yang**](https://z-x-yang.github.io/) · [**Yi Yang<sup>*</sup>**](https://scholar.google.com/citations?user=RMSuNFwAAAAJ)

ReLER, CCAI, Zhejiang University 

<sup>*</sup>corresponding authors

<a href='https://zhenglinzhou.github.io/HeadStudio-ProjectPage/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04681.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![GitHub](https://img.shields.io/github/stars/ZhenglinZhou/HeadStudio?style=social)](https://github.com/ZhenglinZhou/HeadStudio/)

https://github.com/ZhenglinZhou/HeadStudio/assets/42434623/19893d52-8fe5-473d-b5c0-aea29d6be21a

</div>

## Text to Head Avatars Generation

<p align="center">
<img src="./assets/teaser.png">
</p>

Text-based animatable avatars generation by **HeadStudio**.

## Installation
All the followings have been tested successfully in **cuda 11.8**.
```bash
# clone the github repo
git clone https://github.com/zhenglinzhou/HeadStudio-open.git
cd HeadStudio-open
```

Create a conda environment:
```bash
# make a new conda env (optional)
conda create -n headstudio python=3.9
conda activate headstudio
```

It may take some time to install:
```bash
# install necessary packages
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install some packages using conda
bash packages.sh

# install packages using pip
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

* HeadStudio is built on the FLAME. Before you continue, please kindly register and agree to the license from https://flame.is.tue.mpg.de.
* Download `FLAME 2020` which contains `FLAME_FEMALE.pkl`, `FLAME_GENERIC.pkl`, `FLAME_MAKE.pkl` from https://flame.is.tue.mpg.de.
* Download other ckpts and training/validation files from [here](https://pan.baidu.com/s/1BdFmOMNT4gWhqUKFuZWx9A?pwd=pkwj).
* Make the folder like this:
```
.
|-ckpts
    |-ControlNet-Mediapipe
        |-flame2facemsh.npy
        |-mediapipe_landmark_embedding.npz
    |-FLAME-2000
        |-FLAME_FEMALE.pkl
        |-FLAME_GENERIC.pkl
        |-FLAME_MAKE.pkl
        |-flame_static_embeddings.pkl
        |-flame_dynamic_embeddings.pkl
|-talkshow
    # for training with animation
    |-collection
        |-cemistry_exp.npy
    # for evaluation
    |-ExpressiveWholeBodyDatasetReleaseV1.0
...
```
* Specify the `talkshow_train_path` and `talkshow_val_path` in `./configs/headstudio.yaml`.

## Usage

```bash
python3 launch.py \
--config configs/headstudio.yaml --train system.prompt_processor.prompt='a DSLR portrait of Joker in DC, masterpiece, Studio Quality, 8k, ultra-HD, next generation' \
system.guidance.use_nfsd=True system.max_grad=0.001 system.area_relax=True
```

More examples can be found in `./scripts/headstudio.sh`


## Prepare Animation Data
1. Install [TalkSHOW](https://github.com/yhw-yhw/TalkSHOW). You had better use another python environment for following animation, since TalkSHOW needs python 3.7.

please remember to install `torchaudio~=0.13.1`, `torchvision~=0.14.1`.
 
2. Download SHOW_dataset_v1.0.zip following [this](https://github.com/yhw-yhw/TalkSHOW?tab=readme-ov-file#2-get-data).


## Animation
### Video-based Animation
Animate the avatar using .pkl file captured from video clip (SHOW_dataset_v1.0.zip).
```shell
python3 animation.py
```
### Audio-based Animation
* Copy the ./scripts/demo.py into TalkSHOW folder. 
* Specify the `save_root` in demo.py.
* Given an audio clip, generate FLAME sequences via TalkSHOW as below, please specify `path-to-wav-file`.
```shell
cd TalkSHOW
python3 demo.py \
--config_file ./config/body_pixel.json --infer --audio_file path-to-wav-file \
--id 0 --only_face
```

* Animate avatars using generated FLAME sequences via TalkSHOW.
```shell
python3 animation_TalkSHOW.py --audio path-to-audio --avatar path-to-avatar
```

### Text-based Animation
* Generate the audio with given text using [PlayHT](https://play.ht/). 
* Transfer to audio-based animation.

## Acknowledgements
- HeadStudio is developed by ReLER at Zhejiang University, all copyright reserved.
- Thanks [Duochao](https://github.com/dc-walker) and [Xuancheng](https://github.com/Maplefaith) to fix bugs and further develop this work.
- Thanks [PlayHT](https://play.ht/), we use it for text to audio generation.
- Thanks [TalkSHOW](https://arxiv.org/pdf/2212.04420.pdf), we use it for audio-based avatar driven.
- Thanks [threestudio](https://github.com/threestudio-project/threestudio), [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars/tree/main), [HumanGaussian](https://github.com/alvinliu0/HumanGaussian), [TADA](https://github.com/TingtingLiao/TADA), this work is built on these amazing research works.

## Notes
* If you have questions or find bugs, feel free to open an issue or email the first author (zhenglinzhou@zju.edu.cn)!

## Cite
If you find HeadStudio useful for your research and applications, please cite us using this BibTeX:

```bibtex
@inproceedings{zhou2024headstudio,
  title = {HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting},
  author = {Zhenglin Zhou and Fan Ma and Hehe Fan and Zongxin Yang and Yi Yang},
  booktile = {ECCV},
  year={2024},
}
```