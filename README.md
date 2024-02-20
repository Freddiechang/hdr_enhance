# Lighting Enhancement Using Self-attention Guided HDR Reconstruction
A self-attention based learning strategy inspired by high dynamic range (HDR) reconstruction process to reconstruct a properly exposed image from a single input image.
- This is the first work to utilize self-attention mechanism to model long-distance dependencies across different regions in images for lighting enhancement. This mechanism helps reduce artifacts and boosts the output image quality.
- We design a new HDR loss function inspired by the characteristics of HDR images and the HDR reconstruction process. We show that this loss function can alleviate the color shift/artifacts in the output images.
- We compare our work with several state-of-the-art methods utilizing objective tests to show that our proposed method outperforms all other existing methods.

![Comparison Results](/results.png "Comparison Results")
# Dependencies
```
pytorch
torchvision
tqdm
```
# File Organization
```
.
├── data (dataset handlers)
├── loss (loss functions)
├── model (model definitions)
├── template.py
├── trainer.py
├── option.py (command line options)
├── utility.py
└── main.py
```
# Citation
```
@InProceedings{10.1007/978-3-031-22061-6_30,
author="Zhang, Shupei
and Hu, Kangkang
and Zhou, Zhenkun
and Basu, Anup",
editor="Berretti, Stefano
and Su, Guan-Ming",
title="Lighting Enhancement Using Self-attention Guided HDR Reconstruction",
booktitle="Smart Multimedia",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="409--418",
isbn="978-3-031-22061-6"
}
```
