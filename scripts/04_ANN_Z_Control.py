"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
from pickle import FALSE
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pandas as pd
import pybullet as p
import matplotlib.pyplot as plt
import json
import trajectories

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import tensorflow as tf
assert (tf.__version__=='2.3.0'), 'Versión incorrecta de Tensorflow, por favor instale 2.3.0'

#rpm_list = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
rpm_list = ['RPM0']

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=2,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=10,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--vel_ctrl',           default=True,       type=str2bool,      help='Whether to use Speed Controller (default: False)', metavar='')
    ARGS = parser.parse_args()

    order = 3
    #for filename in os.listdir(directory):
    #for _ in range(1):   

    H = 5
    H_STEP = .05
    INIT_XYZS = np.array([[i, i, H] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0, 0] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

     #### Initialize trajectory ######################
    NUM_WP = ARGS.control_freq_hz*ARGS.duration_sec
    wp_counters = np.array([0]*ARGS.num_drones)
    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_VEL = np.zeros((NUM_WP,3))
    TARGET_RPY = np.zeros((NUM_WP,3))
    TARGET_RPY_RATES = np.zeros((NUM_WP,3))
    TARGET_TRAY_ROT = np.zeros((NUM_WP,3))
    TARGET_TRAY_LIN = np.zeros((NUM_WP,3))
    
    N = 5
    for k in range(10*N):
        print(k)
        TARGET_TRAY_LIN = np.zeros((NUM_WP,3))
        Rand_1 = 2*np.random.rand()
        Rand_2 = 10*np.random.rand()
        if k%N==0:
            z = trajectories.step(NUM_WP, Rand_1)
        elif k%N==1:
            z = trajectories.ramp(NUM_WP, Rand_1, ARGS.control_freq_hz)
        elif k%N==2:
            z = trajectories.square(NUM_WP,Rand_1/20, ARGS.control_freq_hz)
        elif k%N==3:
            z = trajectories.sin(NUM_WP, Rand_1, Rand_2, ARGS.control_freq_hz)
        else:
            z =  trajectories.noise(NUM_WP, ARGS.control_freq_hz, 2*(Rand_1+0.1), 0.7, 4)

        for i in range(NUM_WP):
            if i>2*ARGS.control_freq_hz and i<NUM_WP-3*ARGS.control_freq_hz:
                TARGET_TRAY_LIN[i, :] = 0, 0, z[i]
            #TARGET_TRAY_ROT[i, :] = 0, 0, 0
            if ARGS.vel_ctrl:
                TARGET_VEL[i, :] = TARGET_TRAY_LIN[i, :] 
                TARGET_RPY_RATES[i, :] = TARGET_TRAY_ROT[i, :] 
                #TARGET_RPY[i, :] = TARGET_TRAY_ROT[i, :] 
            else:        
                TARGET_POS[i, :] = TARGET_TRAY_LIN[i, :]
                TARGET_RPY[i, :] = TARGET_TRAY_ROT[i, :] 
        #with open(os.path.join(directory, filename))
        model = tf.keras.models.load_model('Models/Z_model.h5')
        #### Create the environment with or without video capture ##
        env = CtrlAviary(drone_model=ARGS.drone,
                            num_drones=ARGS.num_drones,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            physics=ARGS.physics,
                            neighbourhood_radius=10,
                            freq=ARGS.simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=ARGS.gui,
                            record=ARGS.record_video,
                            obstacles=ARGS.obstacles,
                            user_debug_gui=ARGS.user_debug_gui
                            )

        #### Obtain the PyBullet Client ID from the environment ####
        PYB_CLIENT = env.getPyBulletClient()

        #### Initialize the logger #################################
        logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                        num_drones=ARGS.num_drones
                        )

        #### Initialize the controllers ############################
        if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            ctrl = [DSLPIDControl(env) for i in range(ARGS.num_drones)]
        elif ARGS.drone in [DroneModel.HB]:
            ctrl = [SimplePIDControl(env) for i in range(ARGS.num_drones)]

        #### Run the simulation ####################################
        CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
        action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
        START = time.time()
            

        vel_Z = [0]*order
        acel_Z = [0]*order
        ref_z = [0]*order
        for i in  range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            
            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:
            #### Replay control for the current way point #############
                for j in range(ARGS.num_drones):
                    #print(f"state= {z_state}")
                    if i> env.SIM_FREQ and j==0:
                        z_state = [obs[str(j)]["state"][12], obs[str(j)]["state"][18], TARGET_VEL[wp_counters[j],2]]
                        z_state+=vel_Z+ acel_Z+ ref_z    
                        RPM_Norm = model.predict([z_state])
                        RPM = []
                        with open('logs/Files/dict_act_norm_Z.json') as file:
                            dict_act = json.load(file)
                        for i, rpm in enumerate(rpm_list):
                            RPM.append(RPM_Norm[0][i]*(dict_act[rpm]["max"]-dict_act[rpm]["min"])+dict_act[rpm]["min"]) 
                        #print(f"action= {RPM}")
                        action[str(j)] = [RPM[0], RPM[0], RPM[0], RPM[0]]
                        vel_Z = [obs[str(j)]["state"][12]] + vel_Z[0:order-1]
                        acel_Z = [obs[str(j)]["state"][18]] + acel_Z[0:order-1]
                        ref_z = [TARGET_VEL[wp_counters[j],2]]+ ref_z[0:order-1]

                    else:
                        action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                    state=obs[str(j)]["state"],
                                                    #target_pos=np.hstack([TARGET_POS[wp_counters[j], :]]),
                                                    target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                    target_vel = TARGET_VEL[wp_counters[j], :],
                                                    target_rpy=INIT_RPYS[j, :] + TARGET_RPY[wp_counters[j], :],
                                                    target_rpy_rates = TARGET_RPY_RATES[wp_counters[j], :],
                                                    vel_ctrl = ARGS.vel_ctrl
                                                    )
                    

                    #### Go to the next way point and loop #####################
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

            #### Log the simulation ####################################
            for j in range(ARGS.num_drones):
                logger.log(drone = j,
                        timestamp =i/env.SIM_FREQ,
                        state = obs[str(j)]["state"],
                        control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], TARGET_VEL[wp_counters[j], :], INIT_RPYS[j, :]+TARGET_RPY[wp_counters[j], :], TARGET_RPY_RATES[wp_counters[j], :]])
                        )

            #### Printout ##############################################
            if i%env.SIM_FREQ == 0:
                env.render()
                
            #### Sync the simulation ###################################
            if ARGS.gui:
                sync(i, START, env.TIMESTEP)

        #### Close the environment #################################
        env.close()

        #### Plot the simulation results ###########################
        if ARGS.plot:
            logger.plot()
