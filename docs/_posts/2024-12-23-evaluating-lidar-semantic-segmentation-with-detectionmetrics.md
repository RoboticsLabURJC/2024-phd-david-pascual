---
title: "Evaluating LiDAR semantic segmentation with DetectionMetrics"
categories:
  - Blog
tags:
  - Semantic segmentation
  - LiDAR
  - DetectionMetrics
toc: true
toc_sticky: true
---

As part of our journey towards an effective on board perception for autonomous driving in unstructured environments, a comprehensive analysis of the state-of-the-art methods in LiDAR semantic segmentation is required. In previous weeks, we started exploring the topic with RandLA-Net and Rellis3D. In this blog post, we show quantitative results for another model (KPConv) and dataset (GOOSE). Furthermore, the metrics shown here have been computed using DetectionMetrics, our open-source software.

# DetectionMetrics and Open3D-ML

Thanks to the latest updates introduced in DetectionMetrics ([PR #246](https://github.com/JdeRobot/DetectionMetrics/pull/246)), we are now able to evaluate different LiDAR semantic segmentation models against different datasets. As of now, we support Rellis3D and GOOSE datasets and have tested different models from [Open3D-ML](https://github.com/isl-org/Open3D-ML) (KPConv and RandLA-Net).

Some rework has been done in the original model classes defined in Open3D to be able to store them as torch scripted models, which is the format accepted by DetectionMetrics. In that way, we can perform inference or evaluate using said models even if we don't have access to the torch model definition, making the tool more flexible. The custom model and dataset definitions are available in [proyecto-GAIA/tree/main/Perception/lidar](https://github.com/RoboticsLabURJC/proyecto-GAIA/tree/main/Perception/lidar). Furthermore, we have had to include in DetectionMetrics the different sampling methods and transformations applied to the input clouds by the models tested. As we include more models in the future, we'll have to adapt or extend these functionalities.

As of now we have tested:

| Model | Loss | Dataset | # Classes |
| --- | --- | --- | --- |
| RandLA-Net | Cross entropy | Rellis3D | 19 |
| RandLA-Net | Weighted cross entropy | Rellis3D | 19 |
| KPConv | Cross entropy | Rellis3D | 19 |
| KPConv | Weighted cross entropy | Rellis3D | 19 |
| RandLA-Net | Weighted cross entropy | GOOSE | 63 |
| KPConv | Weighted cross entropy | GOOSE | 63 |

In terms of methodology, it is worth mentioning that instead of training for a fixed number of epochs, given the variety of models and datasets, we have deemed it more appropriate to set an early stopping policy. More specifically, we perform a validation step after every epoch and stop training after we reach 25 epochs without improving the validation IoU. We keep the weights for the model with the best validation IoU.

Also, all the models trained have been finetuned using as a starting point the weights for [SemanticKITTI](https://www.semantic-kitti.org/) provided by Open3D-ML. One issue that we might need to take a look into in the future is the fact that they are stored as checkpoints and the training “resumes”, which means that e.g. the scheduler gets loaded in the same state that it was when the model was stored. This might not be optimal as we might be using a learning rate too small for learning new classes.

Besides, we have had to modify the last layers of the models provided to be able to load the pre-trained weights for SemanticKITTI and then estimate GOOSE classes, due to the difference in the number of classes.

# Rellis3D

Here are the mIoU and accuracy per class for the Rellis3D dataset. The results reported in the original [Rellis3D paper](https://arxiv.org/pdf/2011.12954) are marked with an asterisk. We have bolded the best values per column, but omitting the paper results.

>&#9888;&#65039;
>While mIoU simply represents the IoU averaged per class, the global accuracy value presented here is the real ratio between the number of correct points estimated and the total number of points in the whole test dataset.

| Model | mIoU (%) | dirt | grass | tree | pole | water | sky | vehicle | object | asphalt | building | log | person | fence | bush | concrete | barrier | puddle | mud | rubble |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandLA-Net | 24.35 | - | 62.16 | **65.40** | 0.00 | 0.00 | - | 7.08 | - | - | - | 0.00 | **82.04** | 1.54 | 60.16 | 50.66 | 11.80 | 0.00 | 0.01 | 0.00 |
| RandLA-Net Weighted | **28.78** | - | 63.70 | 63.91 | **19.68** | 0.00 | - | **22.26** | - | - | - | **5.06** | 78.41 | **1.92** | 65.09 | **51.97** | **16.51** | **3.98** | **4.09** | **6.26** |
| KPConv | 23.06 | - | **65.76** | 63.24 | 0.00 | 0.00 | - | 0.00 | - | - | - | 0.00 | 80.89 | 0.03 | 66.47 | 45.57 | 0.93 | 0.00 | 0.00 | 0.00 |
| KPConv Weighted | 23.71 | - | 57.21 | 63.40 | 11.67 | 0.00 | - | 0.48 | - | - | - | 0.00 | 74.82 | 0.54 | **67.30** | 39.05 | 11.65 | 3.49 | 2.35 | 0.00 |
| SalsaNext* | 43.07 | - | 64.74 | 79.04 | 56.26 | 0.00 | - | 23.12 | - | - | - | 18.76 | 83.17 | 16.13 | 72.90 | 75.27 | 75.89 | 23.20 | 9.58 | 5.01 |
| KPConv* | 19.97 | - | 56.41 | 49.25 | 0.00 | 0.00 | - | 0.00 | - | - | - | 0.0 | 81.20 | 0.40 | 58.45 | 33.91 | 0.00 | 0.00 | 0.00 | 0.00 |

| Model | Accuracy (%) | dirt | grass | tree | pole | water | sky | vehicle | object | asphalt | building | log | person | fence | bush | concrete | barrier | puddle | mud | rubble |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandLA-Net | 78.43 | - | **93.51** | **85.31** | 0.00 | 0.00 | - | 1.13 | - | - | - | 0.00 | 94.10 | 2.97 | 68.31 | 79.79 | 44.80 | 0.00 | 0.06 | 0.00 |
| RandLA-Net Weighted | 80.73 | - | 84.96 | 82.78 | **23.30** | 0.00 | - | **27.25** | - | - | - | **37.36** | 94.59 | **15.84** | 76.61 | 85.45 | **82.14** | 28.65 | 58.81 | **67.68** |
| KPConv | **82.03** | - | 90.04 | 80.02 | 0.00 | 0.00 | - | 0.00 | - | - | - | 0.00 | 93.26 | 0.04 | 78.14 | 77.68 | 3.63 | 0.00 | 0.00 | 0.00 |
| KPConv Weighted | 78.96 | - | 74.98 | 81.88 | 35.74 | 0.00 | - | 0.35 | - | - | - | 0.00 | **95.91** | 5.35 | **79.11** | **89.73** | 67.69 | **68.13** | **70.95** | 0.00 |

Given the mIoU results, we confirm that the models we are training have reasonable performance when compared against the results reported in the paper. For KPConv, it is worth mentioning that the model we have trained performs slightly better than the one presented in the paper. These differences could come from data augmentation policies, the pretrained weights we are using, etc.

Another interesting conclusion we can reach is that using a loss function weighted by the frequency of each class clearly helps improving the results achieved for the less common classes in the dataset (e.g., pole, vehicle, log, rubble, etc.), at the expense of the most common ones (e.g. tree).

Overall RandLA-Net with weighted loss seems to be the best performing model, although it is worth mentioning that, in terms of accuracy, KPConv without the weighted loss has the upper hand. It seems like KPConv has a harder time segmenting the less common classes, while being pretty robust segmenting the most frequent ones, and the addition of the weighted loss does not help alleviate this issue as much as it does in the case of RandLA-Net.


# GOOSE

Regarding the GOOSE dataset, in the following table we show the mIoU and accuracy per class. The results reported in the original [GOOSE paper](https://arxiv.org/pdf/2310.16788) are marked with an asterisk. It is important to note that we report the results grouped by categories as it is done in the paper, but the models have been actually trained with the full 64 classes ontology. E.g. *person* and *rider* classes results have been averaged to form the *human* category.

| Model | mIoU (%) | vegetation | terrain | sky | construction | vehicle | road | object | void | sign | human | water | animal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandLA-Net Weighted | **10.54** | **23.99** | **15.86** | - | **7.83** | **2.86** | **2.47** | **7.82** | **48.95** | **6.14** | **0.08** | 0.00 | 0.00 |
| KPConv Weighted | 2.46 | 15.97 | 6.99 | - | 4.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| PVKD* | 34.32 | 47.40 | 28.69 | - | 37.94 | 40.82 | 36.21 | 20.61 | - | 29.89 | 57.94 | 9.42 | - |
| SPVNAS* | 17.32 | 41.08 | 19.27 | - | 21.27 | 14.58 | 6.38 | 21.29 | - | 12.94 | 19.08 | 0.00 | - |

| Model | Accuracy (%) | vegetation | terrain | sky | construction | vehicle | road | object | void | sign | human | water | animal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandLA-Net Weighted | **67.74** | **60.95** | **50.88** | - | **47.16** | **11.24** | **19.17** | **21.12** | **99.88** | **13.14** | **0.12** | 0.00 | 0.00 |
| KPConv Weighted | 54.92 | 33.38 | 16.46 | - | 10.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Unfortunately, the labels for the GOOSE test set are not available yet, and so a fair comparison against the results reported in their paper for different models is not possible. Our temporary solution has been using the validation set as test only and splitting a new validation dataset. For said split, we have tried to keep a similar ratio of samples with respect to the original validation set (around 10%). Besides, our new validation dataset comprises 7 sequences of around 120 samples each. This sequence selection has been optimized so that they contain the closest class distribution to the train set. These decisions have not been arbitrary but follow the procedure implied for the dataset design in the paper.

GOOSE dataset is even more unbalanced than Rellis3D, given that there are 64 classes available. This makes it a much more challenging dataset as it can be seen in the results reported. Also, it is very clear that this situation is much more disadvantageous for KPConv, which is not able to correctly estimate a single point for many of the classes. The difference is not so dramatically big if we look at the global accuracy percentages, which means that, for the most common classes, the results are much more acceptable than it seems. Qualitative comparisons would be required to further understand these results.

# Next steps
- Train with GOOSE but grouped by categories.
- Report confusion matrices and computational cost.
- Record videos for qualitative comparison.
- Train with a fresh new optimizer state, instead of loading it from the pretrained checkpoint.