---
title: "ROS node for semantic segmentation"
categories:
  - Blog
tags:
  - CARLA
  - ROS
  - Semantic segmentation
toc: true
---

# ROS node for semantic segmentation
To enhance interoperability between our unstructured map and the CARLA simulator, we've developed a Python-based ROS node for semantic segmentation. This node processes input from a valid ROS topic, such as an RGB camera attached to a vehicle in the unstructured map. We followed these tutorials for building ROS workspaces, packages, and nodes:

[Beginner: Client libraries — ROS 2 Documentation: Humble documentation](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries.html)

For semantic segmentation, we've employed a TensorFlow model fine-tuned on the [Rellis3D dataset](https://www.unmannedlab.org/research/RELLIS-3D). More details are available on [Rebeca Villaraso's TFM blog](https://roboticslaburjc.github.io/2024-tfm-rebeca-villaraso/). The model weights are hosted on [Hugging Face](https://huggingface.co/GAIA-URJC/Rellis3D_20Labels_Weights/blob/main/SEM_ACFNET_EFFNET_30.weights.h5). It's important to note that the model definition has been adapted from the [TASM library](https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models).

Due to changes in the `tf.keras.regularizers.L2` definition from TensorFlow 2.3.0 onwards, TASM is incompatible with the latest TensorFlow version. However, we've maintained a working version of the TASM library in [our repository](https://github.com/RoboticsLabURJC/proyecto-GAIA/tree/main/Perception/3_SemanticSegmentation/tensorflow_advanced_segmentation_models). To simplify installation, I've added a `setup.py` file, enabling installation as a pip package:

```python
from setuptools import setup, find_packages

setup(
    name="tensorflow_advanced_segmentation_models",
    version="0.0.0",
    packages=find_packages(include=["tensorflow_advanced_segmentation_models"]),
)
```

Another surprising issue we discovered was that for the model weights to be loaded correctly, a prediction step needs to be done first. This is the only way we've found to properly initialize the model parameters before the loading weights operation.

For launching the ROS node, we provide a launch file that uses input arguments including the model weights (linked previously), the classes definition, the model configuration, and the image topic to which the node will subscribe. Detailed instructions for building the ROS workspace and launching the segmentation node can be found in [Simulation/docs/segmentation_node.md](https://github.com/RoboticsLabURJC/proyecto-GAIA/blob/main/Simulation/docs/segmentation_node.md).

In the following video, we demonstrate our node segmenting the information retrieved from a camera attached to a vehicle in our unstructured map, using rviz:

{% include video id="1aiNCAS5dh8n9b7SLCTa-Y09Pz7cuYBHw" provider="google-drive" %}

For further validation, we demonstrate the results of subscribing to the camera topic in the example rosbag provided by the Rellis3D dataset. Although the RGB image isn't properly decoded for some reason, the semantic segmentation functions correctly:

{% include video id="1XtPFSFaw763v_UCGKwRvQq6y4psPXpCm" provider="google-drive" %}

It's worth noting that this rosbag was originally built using ROS1. We used a Python tool called Rosbags (https://ternaris.gitlab.io/rosbags/index.html) to convert it to ROS2 format.