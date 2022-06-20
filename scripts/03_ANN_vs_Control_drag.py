import argparse
import math
import numpy as np
from matplotlib.gridspec import GridSpec
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
    #parser.add_argument('--physics',            default="pyb_gnd_drag_dw",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=30,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    H=1
    #######################
    ax = 'y'
    trajectories = ['sin', 'cos', '']
    axis=['y', 'x', 'z']
    params = [{'val':0.3, 'f':0.1}, {'val':0.3, 'f':0.1}, {}]
    
    
    window = 64
    feedback = False
    flat = False
    root = '../logs/Datasets'
    dataset =  f'Dataset_Final'
    model_path = f'../Models/Dataset_Final_LSTMCNN_Tuner_4.h5'
    norm_data_path = f"{root}/data_description_{dataset}.csv"
    states_list = ['x', 'y','z','p','q','r','vx','vy','vz',
                   'wp','wq','wr','ax','ay','az','ap','aq',
                   'ar','ux','uy','uz', 'ur']
    rpm_list = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
    if feedback:
        states_list+=rpm_list
    
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [1, -0.2, 0.2, 0.25]
    D_PROB = [1, 1, 1, 1]
    DIST_TIME = 6
    N_DIST = 3#ARGS.duration_sec//2
    dist_params = {
        # 'DIST_STATES' : DIST_STATES,
        # 'D_FACTOR' : D_FACTOR,
        # 'D_PROB': D_PROB,
        # 'DIST_TIME' : DIST_TIME,
        # 'N_DIST' : N_DIST
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
        
        
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    ctrls = ['DSLPID', 'LSTMCNN', 'Referencia']
    axis_ctrl = [('u'+s) for s in axis]
    axis_states = ['x', 'y', 'z', 'p', 'q', 'r']
    axis_states_2 = ['vx', 'vy', 'vz', 'wp', 'wq', 'wr', 'ax', 'ay', 'az', 'ap', 'aq', 'ar']
    loc = 'upper right' if trajectories[0]=='stop' else 'lower right'
    #sig_type = 'círculo con físicas básicas'
    sig_type = 'círculo'
    #sig_type = 'círculo con arrastre'
    
    for axs in axis_states+axis_states_2:
        plt.figure(figsize=(6,4))
        
        if axs == 'r':
            plt.ylabel(r'$\psi$ (rad)')
            plt.title(f'Respuesta ante {sig_type} para el eje'+r'$\psi$')
        else:
            plt.ylabel(f'{axs} (m)')
            plt.title(f'Respuesta ante {sig_type} para el eje {axs}')
        plt.xlabel('Tiempo (s)')
        
        for j, log in enumerate(logger):
            controller = log.getStates(drone=0, states=axis_states+axis_ctrl+['t']+axis_states_2)
            plt.plot(controller['t'], controller[axs], '-', color=colors[j],  linewidth=2)
            
        t = controller['t']
        if axs in axis:   
            plt.plot(controller['t'], controller[axs], '--', color='#ff7f0e')
        plt.legend(ctrls, loc=loc)
        plt.xlim(t[0], t[-1])
        plt.grid()
    
    
    fig = plt.figure(figsize=(6,6))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, :],  projection='3d')
    #ax2 = fig.add_subplot(gs[0, 1])
    #ax3 = fig.add_subplot(gs[1, 1])
    for j, log in enumerate(logger):
        controller = log.getStates(drone=0, states=axis+axis_ctrl+['t'])
        x = controller['x']
        y = controller['y']
        z = controller['z']
        t = controller['t']
        ax1.plot3D(x, y, z, label=ctrls[j], color=colors[j], alpha=0.7);
        #ax2.scatter(x, y, label=ctrls[j], color=colors[j], alpha=0.7, s=1);
        #ax3.plot(t, z, label=ctrls[j], color=colors[j]);
    
    rx = controller['ux']
    ry = controller['uy']
    rz = controller['uz']
    ax1.plot3D(rx, ry, rz, label=ctrls[2], color=colors[2], linestyle='--', alpha=0.7);
    #ax2.scatter(rx, ry, label=ctrls[2], color=colors[2], alpha=0.5, s=1);
    #ax3.plot(t, rz, label=ctrls[2], color=colors[2], linestyle='--');
    
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('z (m)')
    ax1.grid(True)
    ax1.legend(loc='upper right',frameon=True)
    fig.suptitle(f'Respuesta ante {sig_type}')
    
    fig = plt.figure(figsize=(6,6))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:, :])
    for j, log in enumerate(logger):
        controller = log.getStates(drone=0, states=axis+axis_ctrl+['t'])
        x = controller['x']
        y = controller['y']
        z = controller['z']
        t = controller['t']
        ax1.scatter(x, y, label=ctrls[j], color=colors[j], alpha=0.7, s=1);
    
    rx = controller['ux']
    ry = controller['uy']
    rz = controller['uz']
    ax1.scatter(rx, ry, label=ctrls[2], color=colors[2], alpha=0.5, s=1);
    
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.grid(True)
    ax1.legend(loc='upper right',frameon=True)

    fig.suptitle(f'Respuesta ante {sig_type}')
    plt.show()
    input("Press Enter to exit...")