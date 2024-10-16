---
title: "CARLA tutorials and unstructured map"
categories:
  - Blog
tags:
  - CARLA
  - tutorial
toc: true
---
Interesting tutorials:


- [Pygame for vehicle control](https://carla.readthedocs.io/en/latest/tuto_G_pygame/)

- [Retrieve simulation data](https://carla.readthedocs.io/en/latest/tuto_G_retrieve_data/)

- [Traffic manager](https://carla.readthedocs.io/en/latest/tuto_G_traffic_manager/)

- [Change textures through the API](https://carla.readthedocs.io/en/latest/tuto_G_texture_streaming/)

- [Instance segmentation sensor](https://carla.readthedocs.io/en/latest/tuto_G_instance_segmentation_sensor/)

- [Generate maps with OpenStreetMap](https://carla.readthedocs.io/en/latest/tuto_G_openstreetmap/)

# Pygame controller

{% include video id="1P7rbMgyCEPiCcZXuxOJEvmaukRTn3uOE" provider="google-drive" %}

# Autopilot and recording from ego-vehicle and spectator

{% include video id="1CyOHwdJ5D5na-TTRnHYzp5vRjXUebr7H" provider="google-drive" %}

# Custom unstructured map

To create a custom map, I duplicated the BaseMap provided by CARLA. This empty map offers a sky as a starting point. I then crafted a *landscape* in the Unreal editor, adding foliage, rocks, and other elements using CARLA's materials. Once complete, you can launch the simulation and spawn a controllable car and camera. However, this process doesn't generate a road layout or waypoints, and lacks traffic management, spawn points, or autopilot functionality. You can see the result in the video below:

{% include video id="1INPyEZV_L6PUjQrJerUR-xhqnHNymOeo" provider="google-drive" %}

# Create map using OpenStreetMap

Thanks to <https://www.openstreetmap.org/>, you can download a `.osm` file containing all road information and easily convert it to .xodr format (OpenDRIVE) using one of CARLA's provided scripts (`PythonAPI/util/osm_to_xodr.py`). After launching CARLA and starting a simulation, you can dynamically set the OpenStreetMap using the command `python config.py -x /path/to/map.xodr`. The resulting map is fully navigable—autopilot, spawn points, and other features work perfectly. You can see an example in the video below:

{% include video id="1YQ-zhZhwvFNK2VGdVfYwQ5eoTZo1qbBk" provider="google-drive" %}