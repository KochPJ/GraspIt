#!/bin/bash

isaac_sim_version="2023.1.0-hotfix.1"
image_tag="latest"
dockerfile="dockerfile"

usage() {
  echo "Usage: $0 [-s <isaac_sim_version>] [-t <image_tag>] [-d <dockerfile>]"
  echo "  -v    Isaac Sim vesion"
  echo "  -t    Docker-Image-Tag"
  echo "  -d    Path to the Dockerfile"
  exit 1
}

while getopts ":v:t:d:" opt; do
  case "$opt" in
    v)
      isaac_sim_version="$OPTARG"
      ;;
    t)
      image_tag="$OPTARG"
      ;;
    d)
      dockerfile="$OPTARG"
      ;;
    *)
      usage
      ;;
  esac
done

echo "ISAAC_SIM_VERSION: $isaac_sim_version"
echo "IMAGE_TAG: $image_tag"
echo "DOCKERFILE: $dockerfile"

docker build --build-arg ISAAC_SIM_VERSION="$isaac_sim_version" --no-cache -t graspsim:"$image_tag" -f "$dockerfile" .
