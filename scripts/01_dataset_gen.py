import argparse
import numpy as np
import random
from DronePySim import Drone, PybulletSimDrone 

from gym_pybullet_drones.utils.utils import str2bool

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb_gnd_drag_dw",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=100,        type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    parser.add_argument('--save_figure',        default=True,       type=str2bool,      help='Whether to save the figure (default: True)', metavar='')
    ARGS = parser.parse_args()

    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [1, 0.4, 0.4, 1]
    #D_FACTOR = [0.7, 0.3, 0.3, 0.6]
    D_PROB = [0.9, 0.5, 0.5, 0.8]
    DIST_TIME = 9
    N_DIST = ARGS.duration_sec//8
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }
    N_UNIQUE_TRAJ = 8
    N_SAMPLES = 500 #Number of samples per unique type of trajectory
    STEP_EACH_CHANGE = 60
    
    #### Initialize the simulation #############################
    
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    for k in range(N_SAMPLES):
        H = 0
        axis=['x','y','z', 'r']
        #axis=['']
        #random.shuffle(axis)
        #for i in range(np.random.randint(0,len(axis)-1)):
        #    axis.pop()
        path = ""
        params = []
        trajectories = []
        for ax in axis:
            Rand_1 = np.random.rand()+0.5
            Rand_2 = np.random.rand()
            l = np.random.randint(0,N_UNIQUE_TRAJ)
            
            if ax == 'vz':
                K = 6
            elif ax == 'r':
                K = 0.95*np.pi
            elif ax == 'z':
                K = 3
            elif ax == 'y' or ax == 'x':
                K = 0.23
            else:
                K = 3
            
            DC = K if ax!='z' else -0.1
            
            if l==0:
                traj_type = "random_step"
                trajectories.append('random_step')
                if ax == 'z':
                    params.append({'val': K*Rand_1, 'DC': 0, 'each':5*STEP_EACH_CHANGE})
                else:    
                    params.append({'val': K*Rand_1, 'DC': -DC*Rand_2, 'each':6*STEP_EACH_CHANGE})
            elif l==1:
                traj_type = "random_step"
                trajectories.append('random_step')
                if ax == 'z':
                    params.append({'val': (K/2)*Rand_1, 'DC': 0, 'each':5*STEP_EACH_CHANGE})
                else:    
                    params.append({'val': (K/2)*Rand_1, 'DC': -DC/2*Rand_2, 'each':6*STEP_EACH_CHANGE})
            elif l==2:
                traj_type = "random_step"
                trajectories.append('random_step')
                if ax == 'z':
                    params.append({'val': Rand_1, 'DC': 0, 'each':5*STEP_EACH_CHANGE})
                else:    
                    params.append({'val': Rand_1, 'DC': -Rand_1, 'each':6*STEP_EACH_CHANGE})
            elif l==3:
                traj_type = "step_notret0"
                trajectories.append('step_notret0')
                if ax == 'z':
                    params.append({'max': (K/2)*Rand_1, 'min': 0, 'N_cycles': 7, 'each':3*STEP_EACH_CHANGE})
                else:
                    params.append({'max': (K/2)*Rand_1, 'min': -DC/2*Rand_2, 'N_cycles': 7, 'each':4*STEP_EACH_CHANGE})              
            elif l==4:
                traj_type = "step_ret0"
                trajectories.append('step_ret0')
                if ax == 'z':
                    params.append({'max': (K/2)*Rand_1, 'min': 0, 'N_cycles': 6, 'each':8*STEP_EACH_CHANGE})
                else:
                    params.append({'max': (K/2)*Rand_1, 'min': -DC/2*Rand_2, 'N_cycles': 6, 'each':9*STEP_EACH_CHANGE})
            elif l==5:
                traj_type = "big_step_notret0"
                trajectories.append('big_step_notret0')
                if ax == 'z':
                    params.append({'max': K/2*Rand_1, 'min': 0, 'N_cycles': 4, 'each':14*STEP_EACH_CHANGE})
                else:
                    params.append({'max': K/2*Rand_1, 'min': -DC/2*Rand_2, 'N_cycles': 4, 'each':15*STEP_EACH_CHANGE})                
            elif l==6:
                traj_type = "stopped"
                trajectories.append('stopped')
                if ax=='z':
                    params.append({'val': Rand_1})
                else:
                    params.append({'val': 0})
            elif l==7:
                traj_type = "stopped"
                trajectories.append('stopped')
                params.append({'val': DC/4*(Rand_1-0.5)})
                H += 1 if ax=='z' else 0
            else:
                traj_type = f"noise"
                trajectories.append('noise')
                params.append({'val':(Rand_1+0.35), 'f0':(K/12)/(7*Rand_1+1)+0.1, 'order':15})
                H += 1 if ax=='z' else 0
                
            path+=ax+'_'+traj_type+'-'
        path+=' '+str(k)
        pySim = PybulletSimDrone(drone_type=ARGS.drone,
                                num_drones = ARGS.num_drones,
                                physics=ARGS.physics, 
                                gui=ARGS.gui,
                                record_video=ARGS.record_video,
                                plot=ARGS.plot, 
                                save_figure = ARGS.save_figure,
                                user_debug_gui=ARGS.user_debug_gui,
                                aggregate=ARGS.aggregate, 
                                obstacles=ARGS.obstacles,
                                save_data = ARGS.save_data,
                                simulation_freq_hz=ARGS.simulation_freq_hz,
                                control_freq_hz=ARGS.control_freq_hz,
                                console_out=True,
                                dist_params = dist_params,
                                duration_sec=ARGS.duration_sec,
                                data_path = path)
        drones = []
        drones.append(
                Drone(INIT_XYZS=[ 2*(np.random.rand()-0.5),2*(np.random.rand()-0.5),H],
                    INIT_RPYS=[0.5*(np.random.rand()-0.5),0.5*(np.random.rand()-0.5),3*(np.random.rand()-0.5)],
                    control_timestep=pySim.control_timestep,
                    i=0,
                    )
                )
        pySim.setdrones(drones)
        pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
        pySim.runSim()
        print(f'k = {k}')
        print(f'axis = {axis}')
        print(f'trajectories = {trajectories}')
        print(f'params = {params}')
        #
        #input("Press Enter to exit...")