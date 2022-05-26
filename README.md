# pyfu
# Deep Sensor Fusion with Pyramid Fusion Networks for 3D Semantic Segmentation

## Contributors:
Hannah Schieber*, Fabian Duerr*, Torsten Schoen and Jürgen Beyerer

- '*' equal contribution
- e-mail: hannah.schieber[at]fau.de

## Abstract

Robust environment perception for autonomous vehicles is a tremendous challenge, which makes a diverse sensor set with e.g. camera, lidar and radar crucial. In the process of understanding the recorded sensor data, 3D semantic segmentation plays an important role. Therefore, this work presents a pyramid-based deep fusion architecture for lidar and camera to improve 3D semantic segmentation of traffic scenes. Individual sensor backbones extract feature maps of camera images and lidar point clouds. A novel Pyramid Fusion Backbone fuses these feature maps at different scales and combines the multimodal features in a feature pyramid to compute valuable multimodal, multi-scale features. The Pyramid Fusion Head aggregates these pyramid features and further refines them in a late fusion step, incorporating the final features of the sensor backbones. The approach is evaluated on two challenging outdoor datasets and different fusion strategies and setups are investigated. It outperforms recent range view based lidar approaches as well as all so far proposed fusion strategies and architectures.

# Quantitative Results

![image](https://user-images.githubusercontent.com/22636930/170203786-c1c6de02-5314-4275-bd36-ae655670f4b5.png)

# Results on SemanticKitti

![image](https://user-images.githubusercontent.com/22636930/170203890-a4f8568e-f59b-4cea-b70c-5b61e20f0ea5.png)

# Results on PandaSet

![image](https://user-images.githubusercontent.com/22636930/170203942-470d0348-21a9-4557-b1d1-c43d246696c3.png)

# Code

```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Part of the Code is borrowed from these repositories [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS) and  [PSPNet](https://github.com/hszhao/PSPNet). Furthermore, it builds upon the fusion idea of Duerr et. al:

```
@INPROCEEDINGS{9287974,  author={Duerr, Fabian and Weigel, Hendrik and Maehlisch, Mirko and Beyerer, Jürgen}, 
	 booktitle={2020 Fourth IEEE International Conference on Robotic Computing (IRC)},  
	 title={Iterative Deep Fusion for 3D Semantic Segmentation},   
	year={2020},  
	volume={},  number={},  pages={391-397},  doi={10.1109/IRC.2020.00067}
}
```

if you cite our work please also consider the previous approach Iterative Deep Fusion for 3D Semantic Segmentation.

```
paper citation coming soon!
```


