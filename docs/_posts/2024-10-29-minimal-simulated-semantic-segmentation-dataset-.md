---
title: "Minimal simulated semantic segmentation dataset"
categories:
  - Blog
tags:
  - CARLA
  - ROS
  - Semantic segmentation
toc: true
---

In the [previous post](https://roboticslaburjc.github.io/2024-phd-david-pascual/blog/custom-semantic-segmentation-camera-in-carla/), we built a landscape as a static mesh and followed the CARLA tutorial to create a working version of custom semantic segmentation in our unstructured environment.

# Path as a static mesh
The remaining challenge was building a path as a static mesh that fit the landscape mesh. Unreal's landscape tools allow us to draw a spline over a landscape and assign any static mesh (in our case, a dirt road). The landscape deforms to fit the drawn spline as closely as possible, and this deformation is parameterized. A simple tutorial demonstrating this process can be seen in the following video:

{% include video id="8WIWuybAKj4" provider="youtube" %}

Subsequently, both the landscape and the spline-drawn road can be exported as independent static meshes and imported into a new map. For the semantic segmentation to work properly, these meshes must be exported to the corresponding directory in CARLA (as explained in the previous post). In our case, the landscape is classified as grass and the path as dirt, corresponding to the classes _GaiaGrass_ and _GaiaDirt_. We store them in our newly created `Content/Static/GaiaGrass` and `Content/Static/GaiaDirt` folders.

# Custom semantic segmentation
With our landscape and path ready, we add other elements such as water and foliage. Notably, foliage can be procedurally added using Unreal's foliage editor. We use this method to add rocks, trees, bushes, and grass. All static meshes used for foliage and water must be properly stored in their corresponding class directories.

Currently, we are using classes that match the Rellis3D dataset ontology. Here's a list of the elements included in our map:

- **_GaiaGrass_**: Blades of grass and landscape base material; traversable.
- **_GaiaDirt_**: Path; traversable.
- **_GaiaTree_**, **_GaiaBush_**: Non-traversable vegetation.
- **_GaiaRubble_**: Rocks; non-traversable, although some are small enough for the vehicle to drive over.
- **_GaiaWater_**: Lake and puddles (currently only outside the path); the vehicle passes through them as if the material didn't exist.
- **_Sky_**.

In the following video, we showcase our new map with custom semantic segmentation:

{% include video id="1Qw5yxgOeTjlLVV6i0fsuq0m0hqixe93B" provider="google-drive" %}

Please note that the video flickers slightly. This is due to the large number of elements in the map. Further optimization is needed to improve the simulation performance.

# Dataset recording
Now that all key elements in our unstructured environment are available, we can create our first simulated dataset. To do this, we launch the CARLA simulator, spawn an autopiloted vehicle with RGB and semantic segmentation cameras, launch our semantic segmentation conversion node for visualization, and record the data in a ROS bag:

```bash
ros2 bag record -o gaia_sim_dataset \
    /carla/hero/rgb_front/image \
    /carla/hero/semantic_segmentation_front/image \
    /segmentation/rgb/carla/hero/semantic_segmentation_front/image
```

The resulting dataset contains 1,935 frames with a resolution of 1024x768 pixels, recorded at 20 fps, totaling 1 minute and 36 seconds. We provide a small Python script that reads the bag file and stores the recorded data frame by frame ([rosbag_to_dataset.py](https://github.com/RoboticsLabURJC/proyecto-GAIA/blob/main/Simulation/scripts/rosbag_to_dataset.py)). An example is shown in the figure below:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/dataset_example.png" alt="">
