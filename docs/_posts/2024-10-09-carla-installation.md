---
title: "CARLA installation"
categories:
  - Blog
tags:
  - CARLA
  - tutorial
toc: true
---

This document serves as supplementary material for installing CARLA from source. While not a step-by-step guide, it details my personal experience during the installation process, highlighting common issues and their workarounds.

📘 [Guide for building CARLA from source (required for development)](https://carla.readthedocs.io/en/latest/build_linux/)

## Prerequisites

System requirements:

- Ubuntu 18.04
- 130 GB disk space
- 8GB dedicated GPU (recommended)

My system:

- Ubuntu 22.04.4
- 611 GB disk space
- NVIDIA GeForce GTX 1080 Ti (11GB)
- AMD Ryzen 7 2700 Eight-Core Processor
- 32GB RAM

### Software requirements
The software requirements installation process went smoothly.

To clone CARLA's Unreal fork, I followed the process outlined at [https://www.unrealengine.com/en-US/ue4-on-github](https://www.unrealengine.com/en-US/ue4-on-github).

The Unreal Engine 4.26 build was successful, taking approximately 1.5 hours to complete. Here's a screenshot of the Unreal editor to confirm the installation:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/unreal.png" alt="">

It's worth noting that compiling all the required shaders to create a new project from a template took considerable time. The process even stalled once, requiring a restart. On the second attempt, it ran smoothly and opened up within seconds.

## Build CARLA

After cloning CARLA, I switched to the master branch. According to the documentation:

>The `master` branch contains the current release of CARLA with the latest fixes and features.

In this case, it's version 0.9.15.

### Compile Python API client
While attempting to compile the Python API client, I encountered the following error:

```bash
Setup.sh: Extracting boost for Python 3.
Building B2 engine..

A C++11 capable compiler is required for building the B2 engine.
Toolset 'clang' does not appear to support C++11.

> clang++ -x c++ -std=c++11  check_cxx11.cpp
check_cxx11.cpp:14:10: fatal error: 'thread' file not found
#include <thread>
         ^~~~~~~~
1 error generated.

** Note, the C++11 capable compiler is _only_ required for building the B2
** engine. The B2 build system allows for using any C++ level and any other
** supported language and resource in your projects.

You can specify the toolset as the argument, i.e.:
    ./build.sh [options] gcc

Toolsets supported by this script are:
    acc, clang, como, gcc, intel-darwin, intel-linux, kcc, kylix, mipspro,
    pathscale, pgi, qcc, sun, sunpro, tru64cxx, vacpp

For any toolset you can override the path to the compiler with the '--cxx'
option. You can also use additional flags for the compiler with the
'--cxxflags' option.

A special toolset; cxx, is available which is used as a fallback when a more
specific toolset is not found and the cxx command is detected. The 'cxx'
toolset will use the '--cxx' and '--cxxflags' options, if present.

Options:
    --help                  Show this help message.
    --verbose               Show messages about what this script is doing.
    --debug                 Build b2 with debug information, and no
                            optimizations.
    --guess-toolset         Print the toolset we can detect for building.
    --cxx=CXX               The compiler exec to use instead of the detected
                            compiler exec.
    --cxxflags=CXXFLAGS     The compiler flags to use in addition to the
                            flags for the detected compiler.

Failed to build B2 build engine
make: *** [Util/BuildTools/Linux.mk:142: setup] Error 1
```

Apparently, this is a well-known issue (<https://github.com/carla-simulator/carla/issues/6901>). Installing the g++ 12 compiler and recompiling solved the problem.

```bash
sudo apt install g++-12
make PythonAPI
```

Remember to restart your terminal or source your `.bashrc` file after setting the `UE4_ROOT` variable and before compiling the API. Failing to do so may result in the following error:

```bash
CMake Error at /usr/share/cmake-3.22/Modules/CMakeDetermineCCompiler.cmake:49 (message):
  Could not find compiler set in environment variable CC:


  /Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v17_clang-10.0.1-centos7/x86_64-unknown-linux-gnu/bin/clang.
Call Stack (most recent call first):
  CMakeLists.txt:33 (project)

CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
-- Configuring incomplete, errors occurred!
See also "/home/dpascualhe/repos/carla/Build/libosm2dr-build/CMakeFiles/CMakeOutput.log".
make: *** [Util/BuildTools/Linux.mk:157: osm2odr] Error 1
```

To install the CARLA client in a virtual environment, you can use the `.whl` file generated during the compilation process. In my case, the generated wheel file is located at:

`PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl`

As you can see, it's been generated for Python 3.10. While it works for me, I'm unsure why only this version was generated. Here's an installation example:

```bash
dpascualhe@dpascualhe:~/repos/carla$ python3.10 -m venv ~/.venvs/carla
dpascualhe@dpascualhe:~/repos/carla$ source ~/.venvs/carla/bin/activate
(carla) dpascualhe@dpascualhe-optical:~/repos/carla$ pip install PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
Processing ./PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
Installing collected packages: carla
Successfully installed carla-0.9.15
```

### Compile the server
Server compilation (`make launch`) is very time-consuming due to the extensive shader compilation required. The process crashed multiple times, requiring repeated attempts. Eventually, I succeeded in compiling and launching the Unreal Engine editor. Below is a screenshot displaying the `Town10 HD` scenario:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/unreal_carla.png" alt="">

It's important to note that even after the editor launches, shaders and mesh distance fields continue compiling for some time.

To verify that graphical acceleration was working correctly, I ran `nvidia-smi`. The `UE4Editor` process was consuming approximately 4-5GB of my GPU's VRAM.

### Start the simulation
Once the editor launched and all shaders and mesh distance fields were ready, I attempted to run the simulation. However, when I pressed the `PLAY` button, the editor crashed. I discovered that my swap partition was too small for CARLA (<https://github.com/carla-simulator/carla/issues/3398>). After increasing my swap size to 8GB, everything ran smoothly (this likely would have prevented some earlier issues as well). This requirement was surprising, given that I have 32GB of RAM.

With the simulation running, I tested the Python client examples suggested in the documentation. While installing the Python requirements, I encountered an issue with the requested `numpy` version. The installation failed with this error message:

```bash
Collecting numpy==1.18.4 (from -r requirements.txt (line 3))
  Using cached numpy-1.18.4.zip (5.4 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [24 lines of output]
      Running from numpy source directory.
      <string>:461: UserWarning: Unrecognized setuptools command, proceeding with generating Cython sources and expanding templates
      Traceback (most recent call last):
        File "/home/dpascualhe/.venvs/carla/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 353, in <module>
          main()
        File "/home/dpascualhe/.venvs/carla/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
        File "/home/dpascualhe/.venvs/carla/lib/python3.10/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 149, in prepare_metadata_for_build_wheel
          return hook(metadata_directory, config_settings)
        File "/tmp/pip-build-env-es85oo_2/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 373, in prepare_metadata_for_build_wheel
          self.run_setup()
        File "/tmp/pip-build-env-es85oo_2/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 503, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/tmp/pip-build-env-es85oo_2/overlay/lib/python3.10/site-packages/setuptools/build_meta.py", line 318, in run_setup
          exec(code, locals())
        File "<string>", line 488, in <module>
        File "<string>", line 465, in setup_package
        File "/tmp/pip-install-75ajk5k2/numpy_bd9cc42d6dae4f8da4480cc76d55c510/numpy/distutils/core.py", line 26, in <module>
          from numpy.distutils.command import config, config_compiler, \
        File "/tmp/pip-install-75ajk5k2/numpy_bd9cc42d6dae4f8da4480cc76d55c510/numpy/distutils/command/config.py", line 20, in <module>
          from numpy.distutils.mingw32ccompiler import generate_manifest
        File "/tmp/pip-install-75ajk5k2/numpy_bd9cc42d6dae4f8da4480cc76d55c510/numpy/distutils/mingw32ccompiler.py", line 34, in <module>
          from distutils.msvccompiler import get_build_version as get_build_msvc_version
      ModuleNotFoundError: No module named 'distutils.msvccompiler'
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

Apparently, recent changes related to `setuptools` are causing this issue (<https://github.com/numpy/numpy/issues/27405>). None of the suggested workarounds were successful for me, so I opted for a simple solution: I removed the exact version match from the `requirements.txt` file and ran `pip install` after activating my `carla` virtual environment. Here's my modified requirements file:

```
future
numpy
pygame
matplotlib
open3d
Pillow
```

Both the `generate_traffic.py` and `dynamic_weather.py` scripts ran flawlessly. However, I'll keep an eye on potential issues that might arise from the installed numpy version. Here's a screenshot of a running simulation:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/unreal_carla_sim.png" alt="">
