
# FLA-UNet
This repo holds code for [FLA-UNet: Feature-location Attention U-Net for Foveal Avascular Zone Segmentation in OCTA Images]
The source code is available at: [FLA-UNet](https://github.com/LiCao-WHPU/FLA-UNet)

# The overall architecture
Since optical coherence tomography angiography (OCTA) is non-invasive and non-contact, it is widely used in the study of retinal disease detection. As a key indicator for retinal disease detection, accurate segmentation of foveal avascular zone (FAZ) has an important impact on clinical application. Although the U-Net and its existing improvement methods have achieved good performance on FAZ segmentation, their generalization ability and segmentation accuracy can be further improved by exploring more effective improvement strategies. In this paper, a novel improved method named Feature-location Attention U-Net (FLA-UNet) is proposed by introducing new designed feature-location attention blocks (FLABs) into U-Net and using a joint loss function. The FLAB consists of feature-aware blocks and location-aware blocks in parallel, and is embed into each decoder of U-Net to integrate more marginal information of FAZ and strengthen the connection between target region and boundary information. The joint loss function is composed of the cross-entropy loss (CE loss) function and the Dice coefficient loss (Dice loss) function, and by adjusting the weights of them, the performance of the network on boundary and internal segmentation can be comprehensively considered to improve its accuracy and robustness for FAZ segmentation. The qualitative and quantitative comparative experiments on the three datasets of OCTAGON, FAZID and OCTA-500 show that, our proposed FLA-UNet achieves better segmentation quality, and is superior to other existing state-of-the-art methods in terms of the MIoU, ACC and Dice coefficient.

## Installation

To run this project, we suggest using Ubuntu 20.04, PyTorch 1.8.0, and CUDA version higher than 10.2.

## Train

MODE = 'Train'
python train.py

## Test

MODE = 'Test'
python train.py

# Credits
If you find this work useful, please consider citing: [Paper-Link](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1463233/full)