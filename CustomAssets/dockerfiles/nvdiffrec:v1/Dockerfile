# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM $BASE_IMAGE

RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

RUN apt-get update && apt-get install -y --no-install-recommends gcc-12 g++-12


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# additional libraries
RUN pip install ninja imageio imageio-ffmpeg xatlas gdown

ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=g++-12

ENV TORCH_CUDA_ARCH_LIST=8.6

# install nvdiffrast
RUN pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation

# HDR image support
RUN imageio_download_bin freeimage

RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \ 
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev \
    libopenexr-dev 

#RUN apt-get install -y \
#    nvidia-cuda-toolkit \
#    nvidia-cuda-toolkit-gcc

RUN DEBIAN_FRONTEND=noninteractive  apt-get install -yq --no-install-recommends gcc-10 g++-10
ENV CC gcc-10
ENV CXX g++-10
ENV CUDAHOSTCXX g++-10

WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git

WORKDIR /opt/colmap

RUN mkdir /usr/include/opencv4
RUN mkdir build
WORKDIR /opt/colmap/build
RUN cmake  .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
RUN ninja
RUN ninja install

ENV PATH="/usr/local/bin:${PATH}"

CMD ["colmap", "help"]
