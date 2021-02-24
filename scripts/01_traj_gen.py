import os
from pickle import ADDITEMS, FALSE
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
from numpy.lib.function_base import extract
from scipy.signal import chirp
import pybullet as p
import matplotlib.pyplot as plt
import trajectories
import utils 

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

CHIRP_METHOD = ['linear', 'quadratic', 'hyperbolic']
DIST_STATES = {'p':7, 'q':8}
#DIST_STATES = {'Q1':3, 'Q1':4,'Q3':5, 'Q4':6}

D_FACTOR = 0.2
DIST_TIME = 8
DIST_N = 5

N_UNIQUE_TRAJ = 5 #Number of unique trajectories
N_SAMPLES = 12 #Number of samples per unique type of trajectory
STEP_EACH_CHANGE = 100

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=100,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--vel_ctrl',           default=True,       type=str2bool,      help='Whether to use Speed Controller (default: False)', metavar='')
    parser.add_argument('--disturbances',       default=True,       type=str2bool,      help='Whether to disturb the DISTURBANCES states (default: False)', metavar='')
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 25
    H_STEP = .05
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    
    #### Initialize trajectory ######################
    NUM_WP = ARGS.control_freq_hz*ARGS.duration_sec
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])
    TARGET_POS = np.zeros((NUM_WP,3))
    TARGET_VEL = np.zeros((NUM_WP,3))
    TARGET_RPY = np.zeros((NUM_WP,3))
    TARGET_RPY_RATES = np.zeros((NUM_WP,3))
    TARGET_TRAY_ROT = np.zeros((NUM_WP,3))
    TARGET_TRAY_LIN = np.zeros((NUM_WP,3))

    for k in range(N_SAMPLES*N_UNIQUE_TRAJ):
        
        TARGET_TRAY_LIN = np.zeros((NUM_WP,3))
        Rand_1 = np.random.rand()
        Rand_2 = np.random.rand()
        if k%N_UNIQUE_TRAJ==0:
            z = trajectories.step_ret0(NUM_WP, 2.5*Rand_1, -2.5*Rand_1, 3, 2*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==1:
            z = trajectories.step_notret0(NUM_WP, 3*Rand_1, -3*Rand_1, 3, STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==2:
            z = trajectories.chirp(NUM_WP, ARGS.control_freq_hz, Rand_1+0.25, 0.1*(Rand_2+0.1), 25*(Rand_2+0.5), 0.9*NUM_WP, method=random.choice(CHIRP_METHOD))
        elif k%N_UNIQUE_TRAJ==3:
            z = trajectories.triangular_sweep(NUM_WP, ARGS.control_freq_hz, 3*Rand_1, 0.1*(Rand_2+0.1), 15*(Rand_2+0.5))
        else:
            z = trajectories.noise(NUM_WP, ARGS.control_freq_hz, 2*(Rand_1+0.1), 0.9/(Rand_1+1), 20)

        for i in range(NUM_WP):
            TARGET_TRAY_LIN[i, :] = 0, 0, z[i]
            #TARGET_TRAY_ROT[i, :] = 0, 0, 0
            if ARGS.vel_ctrl:
                TARGET_VEL[i, :] = TARGET_TRAY_LIN[i, :] 
                TARGET_RPY_RATES[i, :] = TARGET_TRAY_ROT[i, :] 
                #TARGET_RPY[i, :] = TARGET_TRAY_ROT[i, :] 
            else:        
                TARGET_POS[i, :] = TARGET_TRAY_LIN[i, :]
                TARGET_RPY[i, :] = TARGET_TRAY_ROT[i, :] 
        
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

        #### Run the simulation ####################################\
        CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
        action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
        START = time.time()

        ### Disturbances variables ######
        if ARGS.disturbances:
            l=0
            DIST_EVERY_N_STEPS = np.round(ARGS.duration_sec*env.SIM_FREQ/DIST_N)
            Rand_3 = [np.random.rand()]*len(DIST_STATES)
        for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            #print(f"observation = {obs['0']['state']}")  
            #pos_x, pos_y, pos_z, Q1, Q2, Q3, Q4, roll, pitch, yaw,     9
            #vel_x, vel_y, vel_z, ang_vel_x, ang_vel_y, ang_vel_z,  15
            #acc_x, acc_y, acc_z, ang_acc_x, ang_acc_y, ang_acc_z, 
            #last action = rpm0, rpm1, rpm2, rpm3    

            #### Disturbances ##########################################
            if ARGS.disturbances and ((i%DIST_EVERY_N_STEPS == 20) or l>0):
                l += 1
                for j in range(ARGS.num_drones):
                    ##### Disturbance ###################################
                    for nd, d in enumerate(DIST_STATES):
                        dist = 2*(-0.5+Rand_3[nd])*(obs[str(j)]["state"][DIST_STATES[d]]+1)*D_FACTOR
                        obs[str(j)]["state"][DIST_STATES[d]] += dist
                    
                    if ("p" in DIST_STATES) or ("q" in DIST_STATES) or ("r" in DIST_STATES):
                        quat = utils.euler_to_quaternion(obs[str(j)]["state"][7], #roll
                                                         obs[str(j)]["state"][8], #pitch
                                                         obs[str(j)]["state"][9])#yaw
                        for q in range(len(quat)):
                            obs[str(j)]["state"][q+3] = quat[q]
        
                if l>DIST_TIME*CTRL_EVERY_N_STEPS+1:
                    l=0
                    for d in range(len(DIST_STATES)):
                        Rand_3[d] = np.random.rand()

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

        #### Save the simulation results ###########################
        if ARGS.save_data:
            print(f"*******************{k}*******************")
            logger.save_csv(f"{k}")
            
        #### Plot the simulation results ###########################
        if ARGS.plot:
            logger.plot()
