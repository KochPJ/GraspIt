#!/bin/bash

set -e

gpu_index=$1
container_index=$2
num_scenes=$3

echo "Setting variables..."
# Set to desired Nucleus
omni_server="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/2023.1.0"
if ! [[ -z "${OMNI_SERVER}" ]]; then
	omni_server="${OMNI_SERVER}"
fi
# Set to desired Nucleus username
omni_user="admin"
if ! [[ -z "${OMNI_USER}" ]]; then
	omni_user="${OMNI_USER}"
fi
# Set to desired Nucleus password
omni_password="admin"
if ! [[ -z "${OMNI_PASS}" ]]; then
	omni_password="${OMNI_PASS}"
fi
# Set to "Y" to accept EULA
accept_eula=""
if ! [[ -z "${ACCEPT_EULA}" ]]; then
	accept_eula="${ACCEPT_EULA}"
fi
# Set to "Y" to opt-in
privacy_consent=""
if ! [[ -z "${PRIVACY_CONSENT}" ]]; then
	privacy_consent="${PRIVACY_CONSENT}"
fi
# Set to an email or unique user name
privacy_userid="${omni_user}"
if ! [[ -z "${PRIVACY_USERID}" ]]; then
	privacy_userid="${PRIVACY_USERID}"
fi

echo "Running Isaac Sim container..."
docker run --gpus device=$gpu_index  -e "ACCEPT_EULA=${accept_eula}" --rm --network=host --runtime=nvidia\
	-e "OMNI_USER=${omni_user}" -e "OMNI_PASS=${omni_password}" \
	-e "OMNI_SERVER=${omni_server}" \
    -e "PRIVACY_CONSENT=${privacy_consent}" -e "PRIVACY_USERID=${privacy_userid}" \
	-e CONTAINER_INDEX=$container_index \
	-e NUM_SCENES=$num_scenes \
    -v ~/docker/isaac-sim/kit/cache/Kit:/isaac-sim/kit/cache:rw \
	-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
	-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
	-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
	-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
	-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
	-v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
	-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
	-v ~/docker/isaac-sim/documents:/root/Documents:rw \
	-v ~/OptiSim:/omniverse:rw \
	-v /home/sersandr/share:/share:rw \
	scene_gen:latest

echo "Isaac Sim container run completed!"
