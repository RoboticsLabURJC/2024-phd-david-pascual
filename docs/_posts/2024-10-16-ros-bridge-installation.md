---
title: "ROS bridge and unstructured map"
categories:
  - Blog
tags:
  - CARLA
  - ROS
toc: true
---
Now that the ROS bridge has been succesfully installed, it's time to test it in the unstructured map simulation. However, our custom map lacks a valid road definition, so the provided ROS packages don't work out of the box in our environment.

# Custom road definition

After studying the [OpenDRIVE standard](https://publications.pages.asam.net/standards/ASAM_OpenDRIVE/ASAM_OpenDRIVE_Specification/latest/specification/index.html), we've generated a valid `xodr` file that defines a road with several segments within the unstructured map. The process is still somewhat manual:

1. Check the correspondence between the origin of coordinates in the `xodr`/CARLA and the Unreal coordinates. To do this, simply spawn a vehicle at $(x, y, z) = (0, 0, 0)$ and check its coordinates in the Unreal editor. In our case, when $(x_{CARLA}, y_{CARLA}, z_{CARLA}) = (0, 0, 0)$,  $(x_{Unreal}, y_{Unreal}, z_{Unreal}) = (100, -175, 0)$. Considering that Unreal coordinates appear to be in centimeters and CARLA's are in meters, we can convert between them as:

    $$x_{Unreal} = (x_{CARLA} - 100) / 100$$

    $$y_{Unreal} = -(y_{CARLA} - 175) / 100$$

    $$z_{Unreal} = z_{CARLA} / 100$$

2. Place as many empty actors as needed over the unstructured map landscape's surface and note their coordinates.
3. Convert the Unreal coordinates and build a `xodr` file containing a road that links all of them using sequential straight segments. Use the script provided in `Simulation/utils/unreal_pts_to_xodr_road.py`. To use this script, store the Unreal coordinates in a JSON file following this format:

    ```json
    [[x0, y1, z0], [x1, y1, z1], [x2, z2 y2], ..., [xn, zn, yn]]
    ```

    and run the script as:

    ```bash
    python build_xodr.py \
        --unreal_pts <unreal_coordinates_file.json> \
        --out_fname $CARLA_ROOT/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/<your_unstructured_map_name>.xodr \
        --unreal_origin <x0_unreal,y0_unreal,z0_unreal>
    ```


# Launching ROS bridge

Once your `xodr` file is ready, you can test your ROS bridge. The only required modification is removing the lane invasion sensor from the default `carla_spawn_objects` configuration (located in `ros-bridge/src/carla_spawn_objects/config/objects.json`). Otherwise, it will crash when you move the spawned vehicle. Now we're ready to test the CARLA ROS bridge in our unstructured environment using a simple ego-vehicle example.

1. Start the simulation in the Unreal editor.
2. Launch `carla_ros_bridge_with_example_ego_vehicle`:

    ```bash
    ros2 launch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch.py town:=<your_unstructured_map_name>
    ```

3. Launch `rviz` and set the sensors you want to visualize:

    ```bash
    ros2 run rviz2 rviz2
    ```


You can see a working example in the video below:

{% include video id="1OISLRF8ZKXhRrFA_VlxqgNnGj1TYjvcq" provider="google-drive" %}
