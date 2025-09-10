#!/bin/bash

echo $CONTAINER_INDEX
cd /omniverse

/isaac-sim/python.sh simulation.py --mode random --config configs/abc_objects.yaml --headless --random_textures --container_index $CONTAINER_INDEX --save_scene --num_scenes $NUM_SCENES
chmod -R 777 temp
chmod -R 777 out 
chmod -R 777 dataset