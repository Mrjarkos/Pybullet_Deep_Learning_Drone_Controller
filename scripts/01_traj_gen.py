import argparse
import numpy as np
from DronePySim import Drone, PybulletSimDrone 

from gym_pybullet_drones.utils.utils import str2bool

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=str,       help='Physics updates (default: PYB)', metavar='')
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
    parser.add_argument('--save_data',          default=True,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    parser.add_argument('--save_figure',        default=True,       type=str2bool,      help='Whether to save the figure (default: True)', metavar='')
    ARGS = parser.parse_args()

    #axis=['vz','vy', 'vx', 'r']
    axis=['vz']
   
    #DIST_STATES = ['vz', 'p', 'q', 'r']
    DIST_STATES = ['vz', 'p', 'q']
    D_FACTOR = [1, 0.2, 0.2]
    D_PROB = [0.8, 0.5, 0.5]
    #D_FACTOR = [1, 0.2, 0.2, 1]
    #D_PROB = [0.8, 0.5, 0.5, 0.8]
    DIST_TIME = 12
    N_DIST = 20
    num_drones = 1
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }

    N_UNIQUE_TRAJ = 9 #Number of unique trajectories
    N_SAMPLES = 17 #Number of samples per unique type of trajectory
    STEP_EACH_CHANGE = 20
    
    #### Initialize the simulation #############################
    H = 50
    H_STEP = .05
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    for k in range(N_SAMPLES*N_UNIQUE_TRAJ):
        params = []
        trajectories = []
        for ax in axis:
            Rand_1 = np.random.rand()
            l = np.random.randint(0,N_UNIQUE_TRAJ)
            
            if ax == 'vz':
                K = 6
            elif ax == 'r':
                K = np.pi
            else:
                K = 3

            if l==0:
                traj_type = "random_step_1"
                trajectories.append('random_step')
                params.append({'val': K*Rand_1, 'DC': -K*Rand_1, 'each':6*STEP_EACH_CHANGE})
            elif l==1:
                traj_type = "random_step_2"
                trajectories.append('random_step')
                params.append({'val': (K/2)*Rand_1, 'DC': -K/2*Rand_1, 'each':6*STEP_EACH_CHANGE})
            elif l==2:
                traj_type = "random_step_3"
                trajectories.append('random_step')
                params.append({'val': Rand_1, 'DC': -Rand_1, 'each':3*STEP_EACH_CHANGE})
            elif l==3:
                traj_type = "step_notret0"
                trajectories.append('step_notret0')
                params.append({'max': (K/2)*Rand_1, 'min': -K/2*Rand_1, 'N_cycles': 7, 'each':3.2*STEP_EACH_CHANGE})
            elif l==4:
                traj_type = "step_ret0"
                trajectories.append('step_ret0')
                params.append({'max': (K/2)*Rand_1, 'min': -K/2*Rand_1, 'N_cycles': 6, 'each':8*STEP_EACH_CHANGE})
            elif l==5:
                traj_type = "big_step_notret0"
                trajectories.append('big_step_notret0')
                params.append({'max': K*Rand_1, 'min': -K*Rand_1, 'N_cycles': 4, 'each':15*STEP_EACH_CHANGE})
            elif l==6:
                traj_type = "stopped_1"
                trajectories.append('stopped')
                params.append({'val': 0})
            elif l==7:
                traj_type = "stopped_2"
                trajectories.append('stopped')
                params.append({'val': K/4*(Rand_1-0.5)})
            else:
                traj_type = f"noise_{l-7}"
                trajectories.append('noise')
                params.append({'val':(Rand_1+0.35), 'f0':(K/10)/(7*Rand_1+1), 'order':15})

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
                                data_path = f'{k}_{str(trajectories)}')
        drones = []
        for i in range(num_drones):
            drones.append(
                        Drone(INIT_XYZS=INIT_XYZS[i],
                            INIT_RPYS=INIT_RPYS[i],
                            control_timestep=pySim.control_timestep,
                            i=i,
                            vel_ctrl = ARGS.vel_ctrl
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
        # input("Press Enter to exit...")