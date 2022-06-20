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


import sys

sys.path.append("./gym-pybullet-drones")

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
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=10,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    Dataset_name='Dataset_final'
    directory = os.path.dirname(os.path.abspath(__file__))+"/../logs/Datasets/"+Dataset_name

    #for filename in os.listdir(directory):
    for _ in range(5):
        filename = random.choice(os.listdir(directory))
        if not filename.endswith(".csv"):
            continue
        
        print(f"filename={filename}")
        df = pd.read_csv(os.path.join(directory, filename))
        actions = df[["RPM0", "RPM1", "RPM2", "RPM3"]]
        #### Initialize the simulation #############################
        H = df.iloc[0, 3]
        H_STEP = .05
        INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
        INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
        AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

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
        
        #### Run the simulation ####################################
        CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
        action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
        START = time.time()
        for i in range(len(df)):
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            
            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:
            #### Replay control for the current way point #############
                for j in range(ARGS.num_drones):
                   action[str(j)] = actions.iloc[i-1].to_numpy()
                    
            #### Log the simulation ####################################
            for j in range(ARGS.num_drones):
                logger.log(drone = j,
                        timestamp =df.iloc[i, 0],
                        state = obs[str(j)]["state"],
                        control= df.iloc[i, 27:].to_numpy()
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
            logger.plot(plot=True)
            log = logger.getStates(0)
            colors = ['#1f77b4', '#ff7f0e']
            ctrls = ['DSLPID', 'Referencia']
            axis_ctrl = ['ux', 'uy', 'uz', 'ur' ]
            axis_states = ['x', 'y', 'z', 'p', 'q', 'r']
            axis_states_2 = ['vx', 'vy', 'vz', 'wp', 'wq', 'wr', 'ax', 'ay', 'az', 'ap', 'aq', 'ar']
            loc = 'upper right'
            #sig_type = 'círculo con físicas básicas'
            sig_type = filename
            #sig_type = 'círculo con arrastre'
            fig, axs = plt.subplots(6, 3, figsize=(18,15), sharex=True)
            axs[0, 0].plot(log['t'], log['x'], label=ctrls[0], color=colors[0])
            axs[0, 0].plot(log['t'], log['ux'], label=ctrls[1], color=colors[1])
            axs[0, 1].plot(log['t'], log['vx'], label=ctrls[0], color=colors[0])
            axs[0, 2].plot(log['t'], log['ax'], label=ctrls[0], color=colors[0])
            
            axs[1, 0].plot(log['t'], log['y'], label=ctrls[0], color=colors[0])
            axs[1, 0].plot(log['t'], log['uy'], label=ctrls[1], color=colors[1])
            axs[1, 1].plot(log['t'], log['vy'], label=ctrls[0], color=colors[0])
            axs[1, 2].plot(log['t'], log['ay'], label=ctrls[0], color=colors[0])
            
            axs[2, 0].plot(log['t'], log['z'], label=ctrls[0], color=colors[0])
            axs[2, 0].plot(log['t'], log['uz'], label=ctrls[1], color=colors[1])
            axs[2, 1].plot(log['t'], log['vz'], label=ctrls[0], color=colors[0])
            axs[2, 2].plot(log['t'], log['az'], label=ctrls[0], color=colors[0])
            
            axs[3, 0].plot(log['t'], log['r'], label=ctrls[0], color=colors[0])
            axs[3, 0].plot(log['t'], log['ur'], label=ctrls[1], color=colors[1])
            axs[3, 1].plot(log['t'], log['wr'], label=ctrls[0], color=colors[0])
            axs[3, 2].plot(log['t'], log['ar'], label=ctrls[0], color=colors[0])
            
            axs[4, 0].plot(log['t'], log['p'], label=ctrls[0], color=colors[0])
            axs[4, 1].plot(log['t'], log['wp'], label=ctrls[0], color=colors[0])
            axs[4, 2].plot(log['t'], log['ap'], label=ctrls[0], color=colors[0])
            
            axs[5, 0].plot(log['t'], log['q'], label=ctrls[0], color=colors[0])
            axs[5, 1].plot(log['t'], log['wq'], label=ctrls[0], color=colors[0])
            axs[5, 2].plot(log['t'], log['aq'], label=ctrls[0], color=colors[0])
            
            axs[5, 0].set_xlabel('Tiempo (s)')
            axs[5, 1].set_xlabel('Tiempo (s)')
            axs[5, 2].set_xlabel('Tiempo (s)')
            
            axs[0, 0].set_ylabel('x (m)')
            axs[0, 1].set_ylabel('vx (m/s)')
            axs[0, 2].set_ylabel(r'$a_x$ ($m/s^2$)')
            axs[1, 0].set_ylabel('y (m)')
            axs[1, 1].set_ylabel(r'$v_y$ ($m/s$)')
            axs[1, 2].set_ylabel(r'a_x ($m/s^2$)')
            axs[2, 0].set_ylabel('z (m)')
            axs[2, 1].set_ylabel(r'$v_z$ (m/s)')
            axs[2, 2].set_ylabel(r'$a_z$ ($m/s^2$)')
            axs[3, 0].set_ylabel(r'$\psi$ (rad)')
            axs[3, 1].set_ylabel(r'$\omega_\psi$ (rad/s)')
            axs[3, 2].set_ylabel(r'$\alpha_\psi$ ($rad/s^2$)')
            axs[4, 0].set_ylabel(r'$\phi$ (rad)')
            axs[4, 1].set_ylabel(r'$\omega_\phi$ (rad/s)')
            axs[4, 2].set_ylabel(r'$\alpha_\phi$ ($rad/s^2$)')
            axs[5, 0].set_ylabel(r'$\theta$ (rad)')
            axs[5, 1].set_ylabel(r'$\omega_\theta$ (rad/s)')
            axs[5, 2].set_ylabel(r'$\alpha_\theta$ ($rad/s^2$)')
            
            axs[0, 0].grid()
            axs[0, 1].grid()
            axs[0, 2].grid()
            axs[1, 0].grid()
            axs[1, 1].grid()
            axs[1, 2].grid()
            axs[2, 0].grid()
            axs[2, 1].grid()
            axs[2, 2].grid()
            axs[3, 0].grid()
            axs[3, 1].grid()
            axs[3, 2].grid()
            axs[4, 0].grid()
            axs[4, 1].grid()
            axs[4, 2].grid()
            axs[5, 0].grid()
            axs[5, 1].grid()
            axs[5, 2].grid()
            
            axs[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=True, ncol=5)

            
            plt.show()