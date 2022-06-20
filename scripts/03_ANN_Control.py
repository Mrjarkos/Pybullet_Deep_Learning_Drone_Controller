import argparse
import numpy as np
import sys
sys.path.append("../gym-pybullet-drones")

from DronePySim import Drone, PybulletSimDrone 
from NNDrone import NNDrone

from trajectories import *
from gym_pybullet_drones.utils.utils import str2bool

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=20,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=False,      type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    
    #axis=['z','y', 'x', 'r']
    # trajectories=['ramp','sin', 'noise', 'step']
    # params=[{'val':0.05}, {'val': 0.2, 'f':0.2}, {'val': 0.3, 'f0':0.2, 'order':15}, {'val': 0.5}]
    #trajectories=['step','step', 'step', 'ramp']
    #params=[{'val':0.1}, {'val': 0.1}, {'val': 0.1}, {'val': 0.1}]
    
    
    # axis=['z','y', 'x']
    # trajectories=['step','step', 'step']
    # params=[{'val':0.1}, {'val': 0.1}, {'val': 0.1}]
    
    axis=['z','y', 'x', 'r']
    trajectories=['step','step', 'step', 'step']
    params=[{'val':0.1}, {'val': 0.1}, {'val': 0.1}, {'val': 0.1}]
    
    # axis=['x', 'y','z']
    # f = 0.25
    # d= 8
    # params=[]
    # trajectories =lemniscate(ARGS.duration_sec*ARGS.control_freq_hz, d, f, ARGS.control_freq_hz, True)
    # trajectories.append(ramp(ARGS.duration_sec*ARGS.control_freq_hz, 0.1, ARGS.control_freq_hz))
    
    # axis=['y', 'x','z']
    # f = 0.1
    # d= 5
    # trajectories =lemniscate(ARGS.duration_sec*ARGS.control_freq_hz, d, f, ARGS.control_freq_hz, True)
    # trajectories.append(exp(ARGS.duration_sec*ARGS.control_freq_hz, 1, 10/ARGS.duration_sec, ARGS.control_freq_hz))
    
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [1, 0.2, 0.2, 0.25]
    D_PROB = [0.5, 0.5, 0.5, 0.5]
    DIST_TIME = 6
    N_DIST = ARGS.duration_sec//4
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }
    #dist_params = {}
    
    window = 64
    feedback = True
    flat = True
    root = '../logs/Datasets'
    #i = 3
    #I = 'LSTMCNN'
    i = 2
    I = 'CLSTM_Tunner_2'
    dataset =  f'Dataset_Final'
    #model_path = f'../Models/{dataset}_{I}_{i}.h5'
    model_path = f'../Models/ANN_feedback/Dataset_Final_ANN_feedback_10.h5'
    #Dataset_Final_LSTM_0.h5
    #Dataset_Final_LSTMCNN_8.h5
    #Dataset_Final_ANN_5.h5
    
    #model_path = f'../Models/{I}/{dataset}_{I}_{i}.h5'
    # model_path_1 = model_path
    #model_path_1 = f'Models/Dataset_XYZ_1_4.h5'
    norm_data_path = f"{root}/data_description_{dataset}.csv"
    states_list = ['x', 'y','z','p','q','r','vx','vy','vz',
                   'wp','wq','wr','ax','ay','az','ap','aq',
                   'ar','ux','uy','uz', 'ur']
    rpm_list = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
    if feedback:
        states_list+=rpm_list
        
    pySim = PybulletSimDrone(drone_type=ARGS.drone,
                            num_drones = 2,
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
    
    H = 50
    drones = []
    
    drone = NNDrone(INIT_XYZS=[0,0,H],
                    INIT_RPYS=[0,0,0],
                    i=0)
    drone.initControl(
                    model_path=model_path,
                    norm_data_path=norm_data_path,
                    input_list=states_list,
                    output_list=rpm_list,
                    window=window,
                    flat=flat
                    )
    drones.append(drone)
    
    drones.append(
                Drone(INIT_XYZS=[0,0,H+1],
                    INIT_RPYS=[0,0,0],
                    control_timestep=pySim.control_timestep,
                    i=1,
                    )
                )
    pySim.setdrones(drones)
    pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
    pySim.runSim()
    input("Press Enter to exit...")