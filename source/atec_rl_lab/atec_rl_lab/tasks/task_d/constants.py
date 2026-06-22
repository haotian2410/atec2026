"""Shared fixed-map local coordinates for Task D RL curriculum interfaces."""

# All coordinates in this file are local to each Isaac Lab env origin.
# The original competition map x coordinate is shifted by +3m in these local coordinates.
FULL_ROBOT_START = (0.0, 0.0)
FULL_BOX_START = (0.0, 1.6)

B2_STANDING_ROOT_Z = 0.58

CLIMB_BOX_TARGET_X = 2.9
CLIMB_BOX_TARGET_Y = 1.6
CLIMB_BOX_HALF_WIDTH_X = 0.03
CLIMB_BOX_HALF_WIDTH_Y = 0.03
CLIMB_BOX_YAW = 0.0
CLIMB_BOX_YAW_TOL = 0.02

PRE_CLIMB_ROBOT_X = 1.9
PRE_CLIMB_ROBOT_Y = 1.6
PRE_CLIMB_ROBOT_HALF_WIDTH_X = 0.05
PRE_CLIMB_ROBOT_HALF_WIDTH_Y = 0.05
PRE_CLIMB_YAW = 0.0
PRE_CLIMB_YAW_RANGE = (-0.08, 0.08)

DROP_ROBOT_X_RANGE = (3.05, 3.25)
DROP_ROBOT_Y_RANGE = (1.50, 1.70)
DROP_ROBOT_Z = 1.55
DROP_ROBOT_YAW_RANGE = (-0.08, 0.08)

B2_STANDING_JOINT_POS = {
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
}
