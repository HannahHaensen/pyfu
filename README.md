# pyfu
# Deep Sensor Fusion with Pyramid Fusion Networks for 3D Semantic Segmentation

## Contributors:
Hannah Schieber*, Fabian Duerr∗, Torsten Schoen and Jürgen Beyerer

* equal contribution
e-mail: hannah.schieber[at]fau.de

## Abstract

Robust environment perception for autonomous vehicles is a tremendous challenge, which makes a diverse sensor set with e.g. camera, lidar and radar crucial. In the process of understanding the recorded sensor data, 3D semantic segmentation plays an important role. Therefore, this work presents a pyramid-based deep fusion architecture for lidar and camera to improve 3D semantic segmentation of traffic scenes. Individual sensor backbones extract feature maps of camera images and lidar point clouds. A novel Pyramid Fusion Backbone fuses these feature maps at different scales and combines the multimodal features in a feature pyramid to compute valuable multimodal, multi-scale features. The Pyramid Fusion Head aggregates these pyramid features and further refines them in a late fusion step, incorporating the final features of the sensor backbones. The approach is evaluated on two challenging outdoor datasets and different fusion strategies and setups are investigated. It outperforms recent range view based lidar approaches as well as all so far proposed fusion strategies and architectures.
