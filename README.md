## Waking-Up

> 大多数人都高估了他们一天能做的事情，但低估了他们一年能做的事情。


## 导言
> 自动驾驶太火？高薪？跃跃欲试，又仅存于想的阶段。动起来，只是看理论，却总也学不会？看不懂，又总没有进度？如果你也有这类问题，那你来看看这个专栏。以实际项目为导向，亲自动手实践，从简单的图像分类、目标检测开始，逐渐学习掌握实例分割、目标检测、车道线检测等进阶技能。学习有回馈、有成就感，你才能继续下去。转行？你就得看看这个。

**基本思路：自动驾驶感知模块的生产流水线，输入+输出**

<br>

- 本地阅读：`docsify serve .`
- 在线阅读：`https://charmve.github.io/autopilot-perception`

# 课程大纲

## 零、感知系统整体概述 （5%）
（框图）

1. 在自动驾驶系统中的位置，上下游
2. 解决什么问题
3. 实现方案

[自动驾驶感知算法实战——感知系统整体概述](./docs/01-感知系统整体概述.md)

## 一、输入：相机+雷达 （10%）
1. 相机
1.1 相机的成像原理
1.2 数字图像处理：去畸变、resize、颜色变换、转柱面、透视变换等（原理及源代码实现）
1.3 实现方案及性能分析：opencv、nvmedia

[自动驾驶感知算法实战1——车载相机及图像处理](./docs/02-车载相机及图像处理.md)

2. 激光雷达
2.1 雷达模块概述
2.2 雷达感知原理
2.3 雷达感知算法概述

[自动驾驶感知算法实战2——激光雷达原理介绍](./docs/03-激光雷达原理介绍.md)

## 二、感知系统任务/目标 （20%）
[自动驾驶感知算法实战3——自动驾驶2D和3D视觉感知算法概述](./docs/04-自动驾驶2D和3D视觉感知算法概述.md)

1. 语义分割：可通行区域检测、障碍物检测、异形物检测

主要模型介绍分析：UNet、SegNet（原理及源代码实现）

[自动驾驶感知算法实战4——语义分割网络详解（DeepLabV3、FCN、UNet等）](./docs/05-语义分割网络详解（DeepLabV3、FCN、UNet等）.md)

2. 目标检测：行人检测、车辆检测、车道线检测、可通行区域检测、多目标跟踪

主要模型介绍分析：Mask R-CNN、Inception v2（原理及源代码实现）

[自动驾驶感知算法实战4——目标检测网络详解（R-CNN、FastR-CNN、YOLO、SSD等）](./docs/06-目标检测网络详解（R-CNN、FastR-CNN、YOLO、SSD等）.md)

3. 目标分类：红绿灯识别、障碍物检测

主要模型介绍分析：AlexNet、VGG、FCN（原理及源代码实现）

[自动驾驶感知算法实战6——目标分类详解（ResNet、VGG、GoogLeNet等）](./docs/07-目标分类详解（ResNet、VGG、GoogLeNet等）.md)


## 三、图像分类（机器学习方法） （15%）（原理及源代码实现）
3.1 数据驱动方法
- 3.1.1 语义上的差别
- 3.1.2 图像分类任务面临着许多挑战
- 3.1.3 数据驱动的方法

[自动驾驶感知算法实战7——数据驱动方法](https://cs231n.github.io/classification/)

3.2 k 最近邻算法
- 3.2.1 k 近邻模型
- 3.2.2 k 近邻模型三个基本要素
- 3.2.3 KNN算法的决策过程
- 3.2.4 k 近邻算法Python实现

[自动驾驶感知算法实战8——k 最近邻算法](https://charmve.github.io/computer-vision-in-action/#/1_理论篇/chapter3_Image-Classification/chapter3.2_knn)

3.3 支持向量机
- 3.3.1 概述
- 3.3.2 线性支持向量机
- 3.3.3 从零开始实现支持向量机
- 3.3.4 支持向量机的简洁实现

[自动驾驶感知算法实战9——支持向量机](https://charmve.github.io/computer-vision-in-action/#/1_理论篇/chapter3_Image-Classification/chapter3.3_支持向量机)

3.4 逻辑回归 LR
- 3.4.1 逻辑回归模型
- 3.4.2 从零开始实现逻辑回归
- 3.4.3 逻辑回归的简洁实现

[自动驾驶感知算法实战10——逻辑回归 LR](https://charmve.github.io/computer-vision-in-action/#/1_理论篇/chapter3_Image-Classification/chapter3.4_Logistic-Regression)

## 四、多传感器融合感知方案详解 （20%）

1. 感知方案：前融合、后融合、中融合

- 1.1 lidar-基于激光雷达进行障碍物检测、分割、分类
- 1.2 相机-红绿灯检测、障碍物检测和分类
- 1.3 radar-基于毫米波传感器进行速度、姿态估计
- 1.4 融合Fusion-前融合、后融合、中融合中两种及以上

[自动驾驶感知算法实战11——多传感器融合感知方案详解](./docs/11-多传感器融合感知方案详解.md)

2. BEV模型

BEV 基于图像/Lidar/多模态数据的3D检测与分割任务

- 2.1 坐标变换
- 2.3 时间同步、时序任务
- 2.4 精度选择
- 2.5 性能分析

[自动驾驶感知算法实战12——BEV 基于图像/Lidar/多模态数据的3D检测与分割任务](./docs/12-基于BEV多模态数据的3D检测与分割任务.md)

3. 发展方向：多模态感知、多任务处理、大模型

[自动驾驶感知算法实战13——自动驾驶感知未来发展方向分享](./docs/13-自动驾驶感知未来发展方向分享.md)

## 五、感知算法模型生产线 （20%）
（闭环框图）

1. 数据选择（数据采集、数据增强）
2. 数据标注
3. 模型训练
4. 模型量化
5. 模型部署
6. 测试与验证

[自动驾驶感知算法实战14——感知算法模型生产线](./docs/14-感知算法模型生产线.md)

## 六、纯视觉感知和雷达方案对比（5%）

成本和效果两个角度，第一性原理

1. 特斯拉方案
2. 非特斯拉方案

[自动驾驶感知算法实战15——纯视觉感知和多传感器融合方案对比](./docs/15-纯视觉感知和多传感器融合方案对比.md)

## 七、总结：如何打造“高可靠、多冗余、可量化、数据驱动的感知系统”（5%）

1. 高可靠：对障碍物、红绿灯的识别精度有保证
2. 多冗余：各个模块相互支撑、非串行
3. 可量化：PRT、仿真场景测试、Profiling
4. 数据驱动（全流程闭环）

[自动驾驶感知算法实战专栏总结：如何打造“高可靠、多冗余、可量化、数据驱动的感知系统”](./docs/16-专栏总结-如何打造“高可靠、多冗余、可量化、数据驱动的感知系统”.md)


<br>

# 面向人群

1. 自动驾驶行业研发相关从业人员；转行？你就得看看这个。
2. 对自动驾驶系统感兴趣，尤其是感知模块，对自动驾驶有相关了解，有数理基础；
3. 对机器人系统有相关实践经验，对感知算法有基本了解；
4. 其他算法从业者，有数理基础；

<br>

# 课后收益

1. 对自动驾驶有更深的理解，尤其是视觉和雷达感知系统；
2. 有较为全面的认识，对感知系统全算法链路有一定了解，能够自己动手开始一些感知系统中的子任务；
3. 动手实现车道线检测、目标识别、可通行区域检测等算法，源代码实现；
3. 对当前自动驾驶行业有更深的了解，抛砖引玉开展相关工作；
4. 了解几种经典的感知算法模型，从实现原理到模型产出；

# 知识星球

前沿工作和技术及时解读，本项目中的相关内容的解答，涉及到的源代码分享。同行业的专精研小伙伴交流群，共享行业信息。
<div align="center">
  <img alt="知识星球：https://user-images.githubusercontent.com/29084184/204234701-729337a2-e2f8-42c2-a716-f7cb8ffdef2e.jpg" src="https://user-images.githubusercontent.com/29084184/204234701-729337a2-e2f8-42c2-a716-f7cb8ffdef2e.jpg" width="48%">
</dev>
