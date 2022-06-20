import argparse
import math
import numpy as np
import sys

from trajectories import *
# import sys
# sys.path.append('../')
from DronePySim import Drone, PybulletSimDrone 

from gym_pybullet_drones.utils.utils import str2bool

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="dyn",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=50,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    H=0.2
    axis=['z']
    trajectories = ['stopp']
    params = [{'val':0.15}]
    
    #trajectories=['step','ramp', 'chirp_amplin', 'chirp_amplin']
    #axis=['y', 'x','z']
    f = 0.25
    d= 10
    #trajectories =lemniscate(ARGS.duration_sec*ARGS.control_freq_hz, d, f, ARGS.control_freq_hz, True)
    #trajectories.append(exp(ARGS.duration_sec*ARGS.control_freq_hz, 8, 10/ARGS.duration_sec, ARGS.control_freq_hz))
    #params=[{'val':0.5}, {'val': 0.1}, {'val': 1, 'f0':0.1, 'f1':10, 'method':'linear', 'phi':0}, {'val': 1, 'f0':0.1, 'f1':10, 'method':'linear', 'phi':90}]
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [0.5, 0.2, 0.2, 0.25]
    D_PROB = [0.8, 0.5, 0.5, 0.8]
    DIST_TIME = 6
    N_DIST = ARGS.duration_sec//6
    num_drones = 1
    dist_params = {
        # 'DIST_STATES' : DIST_STATES,
        # 'D_FACTOR' : D_FACTOR,
        # 'D_PROB': D_PROB,
        # 'DIST_TIME' : DIST_TIME,
        # 'N_DIST' : N_DIST
    }

    pySim = PybulletSimDrone(drone_type=ARGS.drone,
                            num_drones = ARGS.num_drones,
                            physics=ARGS.physics, 
                            gui=ARGS.gui,
                            record_video=ARGS.record_video,
                            plot=ARGS.plot, 
                            save_figure=False, 
                            user_debug_gui=ARGS.user_debug_gui,
                            aggregate=ARGS.aggregate, 
                            obstacles=ARGS.obstacles,
                            save_data = ARGS.save_data,
                            simulation_freq_hz=ARGS.simulation_freq_hz,
                            control_freq_hz=ARGS.control_freq_hz,
                            console_out=True,
                            dist_params = dist_params,
                            duration_sec=ARGS.duration_sec)
    drones = []
    for i in range(num_drones):
        drones.append(
                    Drone(INIT_XYZS=[1.5*i,1.5*i,H+0.5*i],
                        INIT_RPYS=[0,0,0],
                        control_timestep=pySim.control_timestep,
                        i=i,
                        )
                    )
    pySim.setdrones(drones)
    pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
    logger=pySim.runSim()
    print(logger.getStates(0, ['RPM0'])['RPM0'][-1])
    input("Press Enter to exit...")