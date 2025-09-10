import numpy as np


# TODO: put in config file
#!--------Environment Constans--------#
MAX_ENV = 25

#!--------Scene Constans--------#

# Scene shift
SHIFT_OFFSET_X = 40

#!--------Roboter Constans--------#
# Position
INIT_ROBOTER_POS_X = -0.656749090
INIT_ROBOTER_POS_Y = 1.10561121e-03
INIT_ROBOTER_POS_Z = 0.54570382

INIT_ROBOTER_POS = np.array([INIT_ROBOTER_POS_X, INIT_ROBOTER_POS_Y, INIT_ROBOTER_POS_Z])

# Orientation
INIT_ROBOTER_ORI_X = 0
INIT_ROBOTER_ORI_Y = 90
INIT_ROBOTER_ORI_Z = 0

#!--------GridCloner Constans--------#
SPACING = 2
NUM_PER_ROW = 5


#!--------Phase Constans--------#

# Phase 0
T_INCREMENT_SETTLE = 0.005
MAX_SETTLE_TIME = 0.25

# Phase 1
T_INCREMENT_CLOSE = 0.005
MAX_CLOSE_TIME = 0.25

# TODO: put in config file
# Phase 4
MAX_PERIOD = 1

# TODO: put in config file
#!--------Penulum Constans--------#
INIT_SPEED = np.pi/6
