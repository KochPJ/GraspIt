## Installation
The scene sampler currently runs on Isaac-Siom 2023.1.0-hotfix.1. While this is no longer a release version of Isaac-Sim, docker deployment is still possible. Docker installation of Isaac-Sim requires the [NVIDIA Docker Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

````
git pull https://github.com/sersandre/OptiSim.git
docker pull nvcr.io/nvidia/isaac-sim:2023.1.0-hotfix.1

cd scene_sampler
docker build -t scene_sampler .
````
Though release versions of the Isaac-Sim docker require a GPU driver version of 535.129.03, all tests and deployments were done with CUDA 12.2 and driver version 535.54.03. For further details on installing newer versions of the Isaac-Sim docker, please refer to the [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)

It is recommended to run the scene sampler in a virtual environment. All dependencies are listed in requirements.txt:
````
python3 -m venv optisim
source optisim/bin/activate
pip install -r requirements.txt
````
## Usage
Prior to running the scene sampler assets (and optionally textures) must be provided in .usd format. The filepath can be specified during execution of the scene sampler. The directory for assets and textures should be structured like:
````
/assets_root_path
└── assets
    ├── object_name
    │   ├── object_name.usd
    ...
└── textures
    ├── texture1.jpg
    ...

````

After data generation, the dataset will be in the following format:
````
/dataset
└── scene_*
    ├── frame_*
    │   ├── camera_params_*.json
    │   ├── depth.png
    │   ├── rgb_*.png
    │   ├── semantic_segmentation_*.png
    │   └── semantic_segmentation_labels_*.json
    ...
````
