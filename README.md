# UBC-Ovarian-Cancer-Subtype-Classification-and-Outlier-Detection

本项目的目标是对超大尺寸医学影像进行分类，以辅助诊断卵巢组织的恶性肿瘤亚型。在数据预处理阶段，

首先将超大尺寸的医学影像切割成小块，并对其进行尺度缩放，同时移除了背景占用过多的图片实例。

为了减少测试集中不同来源医学处理的颜色差异，使用染色工具对切割后的图片进行上色，生成多种染色版本的图片。

在模型训练方面，针对单个小块图片（包括染色图），使用图像增强技术，并建立了多种分类模型，包括EfficientNet、SerexNext和ResNeSt。

在模型评估阶段，使用Balanced Accuracy作为主要评价指标，并通过集成学习方法对测试集的图片进行切割处理，逐一推理。

最后，将各预测结果求和后通过Softmax取最大值标签作为最终分类结果。
