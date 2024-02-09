<div align="center">
<h1>HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting</h1>

[**Zhenglin Zhou**](https://scholar.google.com/citations?user=6v7tOfEAAAAJ) · [**Fan Ma**](https://flowerfan.site/) · [**Hehe Fan**](https://hehefan.github.io/) · [**Yang Yi<sup>*</sup>**](https://scholar.google.com/citations?user=RMSuNFwAAAAJ)

ReLER, CCAI, Zhejiang University 

<sup>*</sup>corresponding authors

<a href='https://zhenglinzhou.github.io/HeadStudio-ProjectPage/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='./docs/technical-report.pdf'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![GitHub](https://img.shields.io/github/stars/ZhenglinZhou/HeadStudio?style=social)](https://github.com/ZhenglinZhou/HeadStudio/)

https://github.com/ZhenglinZhou/HeadStudio/assets/42434623/19893d52-8fe5-473d-b5c0-aea29d6be21a

</div>

## Text to Head Avatars Generation

<p align="center">
<img src="./assets/teaser.png">
</p>

Text-based animatable avatars generation by **HeadStudio**.

### Comparison with Previous Works

<p align="center">
<img src="./assets/comparison_static_avatar.png">
</p>

Comparison with the text to static avatar generation methods.
HeadStudio excels at producing high-fidelity head avatars, yielding superior results.

<p align="center">
<img src="./assets/comparison_dynamic_avatar.png">
</p>

<p align="center">
<img src="./assets/comparison_dynamic_avatar_2.png">
</p>
Comparison with the text to dynamic avatar generation methods.
HeadStudio provides effective semantic alignment, smooth expression deformation, and real-time rendering.

## Acknowledgements
- HeadStudio is developed by ReLER at Zhejiang University, all copyright reserved.
- Thanks [threestudio](https://github.com/threestudio-project/threestudio), we use it as base framework.
- Thanks [PlayHT](https://play.ht/), we use it for text to audio generation.
- Thanks [TalkSHOW](https://arxiv.org/pdf/2212.04420.pdf), we use it for audio-based avatar driven.

## Cite
If you find InstantID useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{zhou2024headstudio,
  title = {HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting},
  author = {Zhenglin Zhou, Fan Ma, Hehe Fan, Yi Yang},
  journal={arXiv preprint},
  year={2024}
}
```