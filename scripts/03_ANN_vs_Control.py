import argparse
import math
import numpy as np
import sys

from trajectories import *
# import sys
# sys.path.append('../')
from DronePySim import Drone, PybulletSimDrone 
from NNDrone import NNDrone
from trajectories import *
from sys_info import *

from gym_pybullet_drones.utils.utils import str2bool

step_time = 1
if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=20,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=False,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    H=1
    K=0.7
    axis = ['z', 'x', 'y', 'r']
    trajectories=['step', 'step', 'step', 'step']
    params=[{'val':0.1},
            {'val':0.1},
            {'val':0.1},
            {'val':0.1}]
    # ax = 'r'
    # trajectories = ['stope']
    # axis=[ax]
    # params = [{'val': 0.1}]
    
    window = 64
    feedback = False
    flat = False
    root = '../logs/Datasets'
    dataset =  f'Dataset_Final'
    model_path = f'../Models/Dataset_Final_LSTMCNN_Tuner_4.h5'
    #model_path = f'../Models/CLSTM/Dataset_Final_CLSTM_Tuner_1_0.h5'
    #model_path = f'../Models/CLSTM/Dataset_Final_CLSTM_Tuner_1_2.h5'
    #model_path = f'../Models/CLSTM/Dataset_Final_CLSTM_Tuner_2_2.h5'
    #model_path = f'../Models/CLSTM/Dataset_Final_CLSTM_Tuner_0_3.h5'
    norm_data_path = f"{root}/data_description_{dataset}.csv"
    states_list = ['x', 'y','z','p','q','r','vx','vy','vz',
                   'wp','wq','wr','ax','ay','az','ap','aq',
                   'ar','ux','uy','uz', 'ur']
    rpm_list = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
    if feedback:
        states_list+=rpm_list
    
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [1, -0.2, 0.2, 0.25]
    #D_PROB = [0, 1, 1, 0] X,Y
    #D_PROB = [1, 0, 0, 0] #Z
    D_PROB = [0, 0, 0, 0] #r
    #D_PROB = [0.5, 0.2, 0.5, 1] #r
    DIST_TIME = 6
    N_DIST = 2#ARGS.duration_sec//2
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }
    
    
    logger = []
    for i in range(2):
        pySim = PybulletSimDrone(drone_type='cf2x',
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
                                    dist_params=dist_params,
                                    duration_sec=ARGS.duration_sec)
        drones = []
        if i==1:
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
        else:
            drones.append(
                    Drone(INIT_XYZS=[0,0,H],
                        INIT_RPYS=[0,0,0],
                        control_timestep=pySim.control_timestep,
                        i=0,
                        )
                    )
        
        pySim.setdrones(drones)
        
        pySim.initTrajectory(trajectories=trajectories,
                                params=params,
                                axis=axis)
        logger.append(pySim.runSim())
        
        
    colors = ['#1f77b4', '#2ca02c']
    ctrls = ['DSLPID', 'LSTMCNN']
    axis_ctrl = [('u'+s) for s in axis]
    #loc = 'upper right' if trajectories[0]=='stop' else 'lower right'
    #sig_type = 'escalón'
    # sig_type = 'perturbación'
    # plt.figure(figsize=(6,4))
    
    # if ax == 'r':
    #     plt.ylabel(r'$\psi$ (rad)')
    #     plt.title(f'Respuesta ante {sig_type} para el eje'+r'$\psi$')
    # else:
    #     plt.ylabel(f'{ax} (m)')
    #     plt.title(f'Respuesta ante {sig_type} para el eje {ax}')
    # plt.xlabel('Tiempo (s)')
    
    # print(f'************************ {ax} ************************')
    # for j, log in enumerate(logger):
    #     controller = log.getStates(drone=0, states=axis+axis_ctrl+['t'])
        
    #     i = axis.index(ax)
    #     yout= controller[axis[i]]
    #     u = controller[axis_ctrl[i]]
    #     t = controller['t']
        
    #     y_ts, ts = settling_time(yout, t)
    #     tr, tr_i, yr_i, tr_f, yr_f = rise_time(yout, t, u[-1])
    #     os, tp, y_max = overshoot(yout, t)
    #     iste = ISTE(t, yout, u)
    #     e_ss, e_ss_abs = ess(y_ts, u[-1])
        
    #     plt.plot(controller['t'], controller[axis[0]], '-', color=colors[j],  linewidth=2)
        
    #     # proj_point(ts, y_ts,)
    #     # proj_point(tp, y_max, 'os')
    #     #proj_point(tr_i, yr_i)
    #     #proj_point(tr_f, yr_f)
    #     print(f'******Controller = {ctrls[j]}******')
    #     print(f'Ts = {ts-step_time}, Ys = {y_ts}')
    #     print(f'Os = {os}%, Tp = {tp-step_time}, y_max={y_max}')
    #     print(f'Tr = {tr}')
    #     print(f'Tr_i = {tr_i-step_time}, Tr_f = {tr_f-step_time}')
    #     print(f'e_ss_abs = {e_ss_abs}, e_ss ={e_ss}')
    #     print(f'iste = {iste}')
    
    # plt.plot(controller['t'], controller[axis_ctrl[0]], '--', color='#ff7f0e')
    # plt.legend([ctrls[0],  ctrls[1], 'Referencia'], loc=loc)
    # #plt.legend([ctrls[0], 'Referencia'], loc='lower right')
    # plt.xlim(t[0], t[-1])
    # #plt.ylim(min(yout), 1.02*max(yout))
    # plt.grid()
    
#    print(controller['t'].max())
    
    
    # plt.figure(figsize=(10,6))
    # plt.ylabel('x (m)')
    # plt.xlabel('t (s)')
    # plt.plot(controller['t'], controller[axis[1]], 'r-', linewidth=2)
    # plt.plot(controller['t'], controller[axis_ctrl[1]], 'g--')
    # plt.legend([axis[1]]+[axis_ctrl[1]])
    # plt.grid()
    
    # plt.figure(figsize=(10,6))
    # plt.ylabel('y (m)')
    # plt.xlabel('t (s)')
    # plt.plot(controller['t'], controller[axis[2]], 'r-', linewidth=2)
    # plt.plot(controller['t'], controller[axis_ctrl[2]], 'g--')
    # plt.legend([axis[2]]+[axis_ctrl[2]])
    # plt.grid()
    
    # plt.figure(figsize=(10,6))
    # plt.ylabel('r (rad)')
    # plt.xlabel('t (s)')
    # plt.plot(controller['t'], controller[axis[3]], 'r-', linewidth=2)
    # plt.plot(controller['t'], controller[axis_ctrl[3]], 'g--')
    # plt.legend([axis[3]]+[axis_ctrl[3]])
    # plt.grid()
    
    plt.show()

        
    input("Press Enter to exit...")