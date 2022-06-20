import argparse
from pickle import FALSE
import numpy as np
import sys
sys.path.append("../gym-pybullet-drones")

from DronePySim import Drone, PybulletSimDrone 
from NNDrone import NNDrone

from trajectories import *
from gym_pybullet_drones.utils.utils import str2bool

import os
from sys_info import *

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
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
    # dist_params = {
    #     'DIST_STATES' : DIST_STATES,
    #     'D_FACTOR' : D_FACTOR,
    #     'D_PROB': D_PROB,
    #     'DIST_TIME' : DIST_TIME,
    #     'N_DIST' : N_DIST
    # }
    dist_params = {}
    
    window = 64
    feedback = False
    root = '../logs/Datasets'
    #model_root = '../Models/ANN_feedback'
    #model_root = '../Models/LSTM'
    model_root = '../Models/LSTMCNN'
    #model_root = '../Models/CLSTM'
    #model_root = '../Models/ANN'
    flat=False #ANN
    #i = 0
    #I = 'CLSTM_Tunner_1'
    dataset =  f'Dataset_Final'
    
    norm_data_path = f"{root}/data_description_{dataset}.csv"
    results = []
    for subdir, dirs, files in os.walk(model_root):
        for file in files:
        #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            print(filepath)
            if not filepath.endswith(".h5"):
                continue
            
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
                                    save_figure=True, 
                                    user_debug_gui=ARGS.user_debug_gui,
                                    aggregate=ARGS.aggregate, 
                                    obstacles=ARGS.obstacles,
                                    save_data = ARGS.save_data,
                                    simulation_freq_hz=ARGS.simulation_freq_hz,
                                    control_freq_hz=ARGS.control_freq_hz,
                                    console_out=True,
                                    dist_params = dist_params,
                                    duration_sec=ARGS.duration_sec,
                                    data_path=file)
            
            model_path = filepath
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
                            flat=flat,
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
            logger = pySim.runSim()
            c0 = logger.getStates(drone=0, states=states_list+['t'])
            c1 = logger.getStates(drone=1, states=states_list+['t'])
            # iste_x = ISTE(c0['t'], c0['x'], c1['x'])
            # iste_y = ISTE(c0['t'], c0['y'], c1['y'])
            # iste_z = ISTE(c0['t'], c0['z'], c1['z'])
            # iste_r = ISTE(c0['t'], c0['r'], c1['r'])
            iste_x = MSE(c0['x'], c1['x'])
            iste_y = MSE(c0['y'], c1['y'])
            iste_z = MSE(c0['z'], c1['z'])
            iste_r = MSE(c0['r'], c1['r'])
            avg=np.mean([iste_x, iste_y, iste_z, iste_r])
            results.append({"path": filepath,
                            "x':": iste_x,
                            'y': iste_y,
                            'z': iste_z,
                            'r': iste_r,
                            "avg": avg
                            })
    print(results)
    with open(f'{model_root}/results.txt', 'w') as filehandle:
        for listitem in results:
            filehandle.write('%s\n' % listitem)
    input("Press Enter to exit...")