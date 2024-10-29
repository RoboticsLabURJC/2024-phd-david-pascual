---
title: "Custom semantic segmentation camera in CARLA"
categories:
  - Blog
tags:
  - CARLA
  - ROS
  - Semantic segmentation
toc: true
---

Following the detailed tutorial on [Create semantic tags - CARLA Simulator](https://carla.readthedocs.io/en/latest/tuto_D_create_semantic_tags/), we've successfully added new custom tags to CARLA's semantic segmentation camera. In this post, we outline the additional changes required beyond the tutorial's instructions.

# Landscape as a static mesh

One challenge we faced was that only static meshes can be used as segmentation labels. In previous versions of our [unstructured map](https://github.com/RoboticsLabURJC/proyecto-GAIA/tree/main/Simulation/worlds/Unstructured), we built the main terrain as a landscape, making it inaccessible to the segmentation camera. Fortunately, we discovered that the landscape can be exported to FBX format ([How to export landscape and convert to a mesh? - Epic Developer Community Forums](https://forums.unrealengine.com/t/how-to-export-landscape-and-convert-to-a-mesh/211021/4)) and then imported back to CARLA Unreal Editor as a static mesh in a newly created map ([Add a new map - CARLA Simulator](https://carla.readthedocs.io/en/0.9.10/tuto_A_add_map/#import-binaries)). While we couldn't include the previously painted texture for the landscape, we can assign a new one after including the static mesh landscape in the new map.

To avoid interfering with the default CARLA towns, we've built our new labels as GaiaWater, GaiaRock, etc. In each corresponding folder created in the `Content/Static` directory, we've copied and renamed the static meshes we'll use in our map, including the landscape.

# Semantic segmentation camera format

After making the necessary changes in the source code and building our new map using only properly tagged and stored static meshes, the semantic segmentation can now retrieve the ID of each newly created class.

However, the default converter between semantic segmentation camera raw data and RGB image doesn't work correctly for the new labels. To address this issue, we've modified the `SemanticSegmentationCamera` in the [ROS bridge fork](https://github.com/ttgamage/carla-ros-bridge) we are using. Specifically, we've altered the [`get_carla_image_data_array` function](https://github.com/ttgamage/carla-ros-bridge/blob/c596934b430173a5713bc1ac191ff23ae8df9686/carla_ros_bridge/src/carla_ros_bridge/camera.py#L350) to send the raw semantic segmentation data instead of converting it into CityScapesPalette:

```python
    def get_carla_image_data_array(self, carla_image):
        """
        Function (override) to convert the carla image to a numpy data array
        as input for the cv_bridge.cv2_to_imgmsg() function

        We keep the semantic segmentation data raw so that the user can tune the
        visualization mask as needed

        :param carla_image: carla image object
        :type carla_image: carla.Image
        :return tuple (numpy data array containing the image information, encoding)
        :rtype tuple(numpy.ndarray, string)
        """

        # carla_image.convert(carla.ColorConverter.CityScapesPalette)
        carla_image_data_array = numpy.ndarray(
            shape=(carla_image.height, carla_image.width, 4),
            dtype=numpy.uint8, buffer=carla_image.raw_data)

        # return carla_image_data_array, 'bgra8'
        return carla_image_data_array[:, :, 2], 'mono8'  # only R channel contains info
```

To enhance the visualization of the semantic segmentation raw image data, we've created a new ROS node ([conversion_node](https://github.com/RoboticsLabURJC/proyecto-GAIA/blob/main/Simulation/ros_gaia/src/segmentation/segmentation/conversion_node.py)) in our segmentation package. This node converts the raw data into a mask using our custom class and color definitions.

# Result

In the following video, you can see a basic example showcasing the ground truth semantic segmentation containing the following basic classes: sky, grass, trees, bushes, and rocks.

{% include video id="1GspnFQNdI2ZS_0c06H8NKK0H-SqZNE7E" provider="google-drive" %}

Alongside the ground truth stream, we showcase the mask predicted by our [semantic segmentation node](https://roboticslaburjc.github.io/2024-phd-david-pascual/blog/ros-node-for-semantic-segmentation/). Note that the video was captured via remote desktop, resulting in lower frame rates and occasional stuttering.

Now that we can add custom labels, our next step is to agree on a class ontology for our specific application, based on either Rellis3D or GOOSE datasets, and upgrade and refine our new unstructured map accordingly.