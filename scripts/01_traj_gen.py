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

CHIRP_METHOD = ['linear']
DIST_STATES = ['p', 'q', 'vz']
NOISE_STATES = ['vz','p', 'q']
#DIST_STATES = {'Q1':3, 'Q1':4,'Q3':5, 'Q4':6}
STATES_DICT = {"x":0, "y":1, "z":2,
            "Q1":3, "Q2":4, "Q3":5, "Q4":6,
            "p":7, "q":8, "r":9,
            "vx":10, "vy":11, "vz":12,
            "wp":13, "wq":14, "wr":15,
            "ax":16, "ay":17, "az":18,
            "ap":19, "aq":20, "ar":21,
            "RPM0":22, "RPM1":23, "RPM2":24, "RPM3":25,
            "ux":26, "uy":27, "uz":28,
            "uvx":29, "uvy":30, "uvz":31,
            "up":32, "uq":33, "ur":34,
            "uwp":35, "uwq":36, "uwr":37}

D_FACTOR = [0.2, 0.2, 1]
N_FACTOR = [0.05, 0.001, 0.001]
N_PROB = [1, 0.6, 0.3, 0.3]
D_PROB = [0.25, 0.5, 0.75, 0.8]
DIST_TIME = 8
DIST_N = 20

N_UNIQUE_TRAJ = 10 #Number of unique trajectories
N_SAMPLES = 17 #Number of samples per unique type of trajectory
STEP_EACH_CHANGE = 20

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
    parser.add_argument('--duration_sec',       default=100,        type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--vel_ctrl',           default=True,       type=str2bool,      help='Whether to use Speed Controller (default: False)', metavar='')
    parser.add_argument('--disturbances',       default=True,       type=str2bool,      help='Whether to disturb the DISTURBANCES states (default: False)', metavar='')
    parser.add_argument('--noise',              default=False,       type=str2bool,      help='Whether to add Noise to the NOISE states (default: False)', metavar='')
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    H = 50
    H_STEP = .05
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    
    if ARGS.disturbances:
        STEP_EACH_CHANGE *= 1.25

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
        Rand_1 = 1#np.random.rand()+0.25
        Rand_2 = 2*(np.random.rand()-0.5)
        Rand_3 = [0]*len(DIST_STATES)
        Rand_4 = np.random.rand()
        Rand_5 = np.random.rand()
        unstable = False
        print(f'Rand_1 = {Rand_1}, Rand_2 = {Rand_2}')
        if k%N_UNIQUE_TRAJ==0:
            #traj_type = "triangular_sweep"
            #z = trajectories.triangular_sweep(NUM_WP, ARGS.control_freq_hz, 2.5, 0.25, 10)
            traj_type = "step_random"
            z = trajectories.big_step_ret0(NUM_WP, 4*Rand_1, -3*Rand_1, 4, 15*STEP_EACH_CHANGE)
            #z = trajectories.random_step(NUM_WP, 7*Rand_1, -6*Rand_1, 6*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==1:
            traj_type = "step_random"
            z = trajectories.random_step(NUM_WP, 4*Rand_1, -3*Rand_1, 6*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==2:
            traj_type = "step_random"
            z = trajectories.random_step(NUM_WP, Rand_1, -Rand_1, 2*STEP_EACH_CHANGE)+Rand_2
        elif k%N_UNIQUE_TRAJ==3:
            traj_type = "step_notret0"
            z = trajectories.step_notret0(NUM_WP, 5*Rand_1, -4*Rand_1, 7, 3.2*STEP_EACH_CHANGE)+Rand_2
        elif k%N_UNIQUE_TRAJ==4:
            traj_type = "step_ret0"
            z = trajectories.step_ret0(NUM_WP, 4*Rand_1, -3*Rand_1, 6, 8*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==5:
            traj_type = "ramp_step_notret0"
            z = trajectories.ramp_step_notret0(NUM_WP, 4*Rand_1, -3*Rand_1, 4, 11*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==6:
            traj_type = "big_step_notret0"
            z = trajectories.big_step_ret0(NUM_WP, 4*Rand_1, -3*Rand_1, 4, 15*STEP_EACH_CHANGE)
        elif k%N_UNIQUE_TRAJ==7:
            traj_type = "stopped"
            z = trajectories.stopped(NUM_WP, 0)
        elif k%N_UNIQUE_TRAJ==8:
            traj_type = "stopped"
            z = trajectories.stopped(NUM_WP, 2*(Rand_1-0.5))
        else:
            traj_type = "noise"
            z = trajectories.noise(NUM_WP, ARGS.control_freq_hz, 2*(2*Rand_1+0.35), 0.7/(6*Rand_1+1), 15)

        if  k%N_UNIQUE_TRAJ==0:
            traj_type = "sawtooth_sweep"
            z = trajectories.sawtooth_sweep(NUM_WP, ARGS.control_freq_hz, 1, 0.25, 25)
        elif  k%N_UNIQUE_TRAJ==1:
            traj_type = "chirp"
            z = trajectories.chirp(NUM_WP, ARGS.control_freq_hz, 1, 0.25, 40, 0.9*NUM_WP, method='linear')
        elif  k%N_UNIQUE_TRAJ==2:
            traj_type = "triangular_sweep"
            z = trajectories.triangular_sweep(NUM_WP, ARGS.control_freq_hz, 2.5, 0.25, 10)
        else:
            break

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
            DIST_EVERY_N_STEPS = np.ceil(ARGS.duration_sec*env.SIM_FREQ/DIST_N)
            if traj_type == "stopped":
                DIST_EVERY_N_STEPS *= 0.8

        for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
            Rand_4 = np.random.rand()
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            if unstable:
                break
            #### Noise #################################################
            if ARGS.noise and Rand_5<=N_PROB[0]:
                for j in range(ARGS.num_drones):
                    for nd, d in enumerate(NOISE_STATES):
                        if np.random.rand()<=N_PROB[nd+1]:
                            Rand_4=np.random.rand()
                        else:
                            Rand_4=0.5
                        obs[str(j)]["state"][STATES_DICT[d]] += N_FACTOR[nd]*2*(Rand_4-0.5)

                    if ("p" in NOISE_STATES) or ("q" in NOISE_STATES) or ("r" in NOISE_STATES):
                        quat = utils.euler_to_quaternion(obs[str(j)]["state"][7], #roll
                                                         obs[str(j)]["state"][8], #pitch
                                                         obs[str(j)]["state"][9])#yaw
                        for q in range(len(quat)):
                            obs[str(j)]["state"][q+3] = quat[q]
                            
            #### Disturbances ##########################################
            if ARGS.disturbances and ((i%DIST_EVERY_N_STEPS == 20) or l>0):
                l += 1
                for j in range(ARGS.num_drones):
                    if abs(obs[str(j)]["state"][STATES_DICT['p']])>0.7 or abs(obs[str(j)]["state"][STATES_DICT['q']])>0.7 or abs(obs[str(j)]["state"][STATES_DICT['z']])<2 or abs(obs[str(j)]["state"][STATES_DICT['vz']])>12:
                        unstable = True
                        print('***************UNSTABLE************')
                        break
                    
                    ##### Disturbance ###################################
                    for nd, d in enumerate(DIST_STATES):
                        obs[str(j)]["state"][STATES_DICT[d]] += D_FACTOR[nd]*2*(Rand_3[nd]-0.5)
                    
                    if ("p" in DIST_STATES) or ("q" in DIST_STATES) or ("r" in DIST_STATES):
                        quat = utils.euler_to_quaternion(obs[str(j)]["state"][7], #roll
                                                         obs[str(j)]["state"][8], #pitch
                                                         obs[str(j)]["state"][9])#yaw
                        for q in range(len(quat)):
                            obs[str(j)]["state"][q+3] = quat[q]
        
                if l>(DIST_TIME*1.5*(np.random.rand()+0.5))*CTRL_EVERY_N_STEPS+1:
                    l=0
                    Rand_3 = [0.5]*len(DIST_STATES)
                    if Rand_4<D_PROB[0]:
                        Rand_3[0] = np.random.rand()
                    elif Rand_4<D_PROB[1]:
                        Rand_3[1] = np.random.rand()
                    elif Rand_4<D_PROB[2]:
                        Rand_3 = np.random.rand(len(DIST_STATES)).tolist()
                        Rand_3[2] = 0.5

                    if np.random.rand()<=D_PROB[3]:
                        Rand_3[2] = np.random.rand()


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
        if ARGS.save_data and not unstable:
            print(f"*******************{k}_{traj_type}*******************")
            logger.save_csv(f"{k}_{traj_type}")
            
        #### Plot the simulation results ###########################
        if ARGS.plot:
            utils.plot_fourier(z, ARGS.control_freq_hz)#frequency analysis
            logger.plot()            
