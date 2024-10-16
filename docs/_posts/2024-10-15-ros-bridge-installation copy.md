---
title: "ROS bridge installation"
categories:
  - Blog
tags:
  - CARLA
  - ROS
  - tutorial
toc: false
---
The [official documentation](https://carla.readthedocs.io/projects/ros-bridge/en/latest/) only supports ROS 2 Foxy. However, forks of the `ros-bridge` repository are available that support the latest CARLA version (0.9.15) and ROS 2 Humble.

1. Install ROS 2 Humble following the [official instructions](https://docs.ros.org/en/humble/Installation.html). Complete some suggested tutorials to verify the installation.
2. Install `ros-bridge` from the following fork (instructions in the README mirror those in the official documentation):

    [https://github.com/ttgamage/carla-ros-bridge](https://github.com/ttgamage/carla-ros-bridge)


The installation of both ROS Humble and the CARLA ROS bridge fork should proceed smoothly. To validate the installation, follow the instructions in <https://carla.readthedocs.io/projects/ros-bridge/en/latest/ros_installation_ros2/#run-the-ros-bridge> before testing the provided ROS packages.

Key packages that should be functional include:

- <https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_spawn_objects/>
- <https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_manual_control/>
- <https://carla.readthedocs.io/projects/ros-bridge/en/latest/rviz_plugin/>

The video below demonstrates starting a simulation, launching `carla_ros_bridge`, spawning a vehicle with `carla_spawn_objects`, enabling manual control via `carla_manual_control`, and visualizing sensors using `rviz`.

{% include video id="1stXjl1e8Sv-hKWil2TcJ_9L75cEZs6ZB" provider="google-drive" %}