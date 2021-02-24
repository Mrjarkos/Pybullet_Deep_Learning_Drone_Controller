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
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=10,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--vel_ctrl',           default=True,       type=str2bool,      help='Whether to use Speed Controller (default: False)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 4
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize trajectory ######################
    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_VEL = np.zeros((NUM_WP,3))
    TARGET_RPY = np.zeros((NUM_WP,3))
    TARGET_RPY_RATES = np.zeros((NUM_WP,3))
    TARGET_TRAY_ROT = np.zeros((NUM_WP,3))
    TARGET_TRAY_LIN = np.zeros((NUM_WP,3))
    
    directory = os.path.dirname(os.path.abspath(__file__))+"/gym-pybullet-drones/files/logs/Trajectories/"
    for t in directory:
        TARGET_TRAY_LIN = np.zeros((NUM_WP,3))
        file_x = random.choice(os.listdir(directory))
        file_y = random.choice(os.listdir(directory))
        file_z = random.choice(os.listdir(directory))
        x = np.load(os.path.join(directory, file_x))
        y = np.load(os.path.join(directory, file_y))
        z = np.load(os.path.join(directory, file_z))
        
        for i in range(NUM_WP):
            TARGET_TRAY_LIN[i, :] = x[i], y[i], z[i]
            #TARGET_TRAY_ROT[i, :] = 0, 0, 0
        #with open(os.path.join(directory, filename))

        if ARGS.vel_ctrl:
            for i in range(NUM_WP):
                TARGET_VEL[i, :] = TARGET_TRAY_LIN[i, :] 
                TARGET_RPY_RATES[i, :] = TARGET_TRAY_ROT[i, :] 
        else:        
            for i in range(NUM_WP):
                TARGET_POS[i, :] = TARGET_TRAY_LIN[i, :]
                TARGET_RPY[i, :] = TARGET_TRAY_ROT[i, :] 
        
        wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])
        #### Create the environment with or without video capture ##
        if ARGS.vision: 
            env = VisionAviary(drone_model=ARGS.drone,
                            num_drones=ARGS.num_drones,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            physics=ARGS.physics,
                            neighbourhood_radius=10,
                            freq=ARGS.simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=ARGS.gui,
                            record=ARGS.record_video,
                            obstacles=ARGS.obstacles
                            )
        else: 
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
        for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            #print(f"observación = {obs['0']['state']}")  
            #pos_x, pos_y, pos_z, Q1, Q2, Q3, Q4, roll, pitch, yaw, 
            #vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z, 
            #acc_x, acc_y, acc_z, ang_acc_x, ang_acc_y, ang_acc_z, 
            #last action = rpm0, rpm1, rpm2, rpm3

            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:

                #### Compute control for the current way point #############
                for j in range(ARGS.num_drones):
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
                for j in range(ARGS.num_drones): 
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

            #### Log the simulation ####################################
            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state= obs[str(j)]["state"],
                        #control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                        #control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                        control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], TARGET_VEL[wp_counters[j], :], INIT_RPYS[j, :]+TARGET_RPY[wp_counters[j], :], TARGET_RPY_RATES[wp_counters[j], :]])
                        )

            #### Printout ##############################################
            if i%env.SIM_FREQ == 0:
                env.render()
                #### Print matrices with the images captured by each drone #
                if ARGS.vision:
                    for j in range(ARGS.num_drones):
                        print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                            obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                            obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                            )

            #### Sync the simulation ###################################
            if ARGS.gui:
                sync(i, START, env.TIMESTEP)

        #### Close the environment #################################
        env.close()

        #### Save the simulation results ###########################
        logger.save_csv(f"{file_x[:len(file_x)-4]}+{file_y[:len(file_y)-4]}+{file_z[:len(file_z)-4]}")
        print(f"*******************{k}*******************")
        #### Plot the simulation results ###########################
        if ARGS.plot:
            logger.plot()
