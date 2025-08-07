## Overview

<img src="images/grasp_clone.png" width="330"/> <img src="images/grasp_cup.png" width="330"/> <img src="images/grasp_bottle.png" width="330"/>

## Installation
The robot simulation currently runs on Isaac-Sim 2023.1.0-hotfix.1. While this is no longer a release version of Isaac-Sim, docker deployment is still possible. Docker installation of Isaac-Sim requires the [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

````
git pull https://github.com/sersandre/OptiSim.git
docker pull nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1

cd graspsim/docker
./build_docker.sh
````

## Usage