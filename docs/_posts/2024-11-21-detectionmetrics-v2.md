---
title: "DetectionMetrics v2"
categories:
  - Blog
tags:
  - Semantic segmentation
  - DetectionMetrics
toc: true
toc_sticky: true
---

Our current project requires a tool to efficiently evaluate image and LiDAR segmentation models, one that's compatible with the most common datasets and deep learning frameworks. DetectionMetrics [[1]](#references), developed by RoboticsLabURJC members, serves a similar purpose but focuses on object detection tasks. As stated on the project webpage:

> Detection Metrics is an application that provides a toolbox of utilities oriented to simplify the development and testing of solutions based on object detection. The application comes with a GUI (based on Qt) but it can also be used through command line.
>

Initially, we considered extending DetectionMetrics to meet these needs. However, modernizing the codebase and adapting it for our purposes would require too much effort. Instead, we decided to build a streamlined version almost from scratch that focuses on our specific requirements.

With that goal in mind, we have built ***DetectionMetrics v2***. It is a library written in Python with a focus on interoperability between different datasets and deep learning frameworks. It is easily extensible and as of now it supports:

- **Datasets**: Rellis3D, GOOSE and GAIA (our custom format)
- **Deep-learning frameworks**: TensorFlow and PyTorch.

# Structure

The library consists of three main components:

- **Datasets**: Contains a parent `ImageSegmentationDataset` class and child objects for each supported dataset. Internally, datasets store a pandas *DataFrame* that links pairs of label and image filenames with their sample IDs and splits, along with a dictionary containing the dataset's ontology. After reading a dataset, we can store it in our custom GAIA format (the *DataFrame* in a Parquet file plus the image files). The implementation also supports merging datasets that have compatible ontologies or by using an *ontology translation* file.
- **Models**: Contains a parent `ImageSegmenationModel` class and child objects for each supported dataset. Each model requires three files: the model file itself, an ontology file, and a configuration file that specifies parameters like image size and normalization values for different frameworks. Once loaded, a model can perform single-image inference or evaluate entire datasets (provided the model and dataset ontologies are compatible). Evaluation results are returned as a *DataFrame* that can be exported to various formats (e.g., CSV).
- **Utils**: Contains miscellaneous utility functions. Notable among these is `utils.metrics`, which defines the metrics used for model evaluation (e.g., IoU).

We also provide examples for loading the compatible datasets and performing evaluation with PyTorch and TensorFlow models in the [examples](https://github.com/JdeRobot/DetectionMetrics/tree/revert-232-revert-231-dph/v2/examples) directory.

# Future work

In the near future, we plan to expand our available metrics, add ONNX support, and incorporate LiDAR datasets. We also aim to develop a GUI for exploring loaded datasets and evaluation results. On the infrastructure side, we will implement testing and set up automated deployment of our library as a PIP package, with automatic documentation updates.

# References

[1] [S. Paniego, V. Sharma, J. Cañas. "Open Source Assessment of Deep Learning Visual Object Detection," in Sensors, vol. 22, no. 12, 2022.](https://www.mdpi.com/1424-8220/22/12/4575)