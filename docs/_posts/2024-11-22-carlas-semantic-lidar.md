---
title: "CARLA’s semantic LiDAR"
categories:
  - Blog
tags:
  - Semantic segmentation
  - CARLA
  - LiDAR
toc: true
toc_sticky: true
---

In addition to CARLA's standard LiDAR sensor, a version that provides semantic information for each point is available. This sensor's frame classes match those encoded in the image segmentation sensor. More info on customizing semantic data in CARLA is available in [previous posts](https://roboticslaburjc.github.io/2024-phd-david-pascual/blog/custom-semantic-segmentation-camera-in-carla/).

# Sensor definition

The semantic LiDAR provides the following data for each point:

1. **Position**: \\((X, Y, Z)\\)
2. **CosAngle**: possibly related to the surface normal at the LiDAR collision point
3. **ObjIdx**: unique identifier for each object instance
4. **ObjTag**: unique identifier for each class

# Data retrieval

During our initial LiDAR data retrieval tests, we found that the sensor ignored certain static meshes. Further investigation revealed that these ignored objects were those without collision configuration—specifically grass elements that were meant to be traversable.

Fortunately, Unreal's flexible collision settings allow different collision types for various map elements. We configured the problematic meshes as solid objects that vehicles could pass through, enabling the sensor to detect them while allowing cars to drive over them naturally.

While using a custom ROS node to store LiDAR frames for processing, we encountered frame flickering and synchronization issues. Though we haven't fully investigated the cause, we found that the Python API provides more reliable control over the simulation and simplifies data stream parsing.

Based on [Félix Martínez's LiDAR visualization tool](https://github.com/RoboticsLabURJC/2024-tfg-felix-martinez/tree/main/Lidar-Visualizer), we developed a script that:

1. Starts a CARLA client
2. Spawns a vehicle with a semantic LiDAR attached and sets the autopilot mode
3. Displays a real-time RGB camera view from a third-person perspective
4. Decodes the LiDAR frame and assigns the corresponding RGB value for each label
5. Saves the decoded semantic LiDAR frames to a specified directory in PLY format, preserving each point's position and RGB values.

The script is available at [`Simulation/scripts/lidar_to_dataset.py`](https://github.com/RoboticsLabURJC/proyecto-GAIA/blob/main/Simulation/scripts/lidar_to_dataset.py).

# Result

The recorded frames can be visualized using [Open3D](https://www.open3d.org/). The following video demonstrates a complete lap in our unstructured map with the recorded LiDAR:

{% include video id="1w9a9GA2Dca2q3yVuuoXkV4fm2CJSA3L9" provider="google-drive" %}
