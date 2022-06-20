import argparse
import numpy as np
from DronePySim import Drone, PybulletSimDrone 
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.utils import str2bool

if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
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
    parser.add_argument('--duration_sec',       default=50,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=False,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    parser.add_argument('--save_figure',        default=False,       type=str2bool,      help='Whether to save the figure (default: True)', metavar='')
    ARGS = parser.parse_args()
    
    #### Initialize the simulation #############################
    H = 0.2
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    j= 3
    K = 0.7
    STEP_EACH_CHANGE = 350
    axis = ['z', 'x', 'y']
    trajectories=['noise', 'step_ret0', 'step_notret0']
    params=[{'val':(0.2+0.35), 'f0':(K/12)/(7*0.3+1)+0.1, 'order':15},
            {'max': K/3, 'min':0, 'each': 210, 'N_cycles':1},
            {'max': 0, 'min':-K, 'each': 250, 'N_cycles':1}]
    
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = [1, 0.2, 0.2, 0.25]
    D_PROB = [0.6, 0.6, 0.6, 1]
    DIST_TIME = 6
    N_DIST = ARGS.duration_sec//10
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }
    path = []
    for i in range(len(trajectories)):
        path.append(axis[i]+'_'+trajectories[i])
    path='-'.join(path)+'-'+str(K)+' '+str(H)+' '+str(j)
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
                            data_path = f'{path}')
    drones = []
    drones.append(
                Drone(INIT_XYZS=[0,0,H],
                    INIT_RPYS=[0,0,0],
                    control_timestep=pySim.control_timestep,
                    i=0,
                    )
                )
    pySim.setdrones(drones)
    pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
    logger = pySim.runSim()
    
    print(f'axis = {axis}')
    print(f'trajectories = {trajectories}')
    print(f'params = {params}')
    
    
    log = logger.getStates(0)
    colors = ['#1f77b4', '#ff7f0e']
    ctrls = ['DSLPID', 'Referencia']
    axis_ctrl = ['ux', 'uy', 'uz', 'ur' ]
    axis_states = ['x', 'y', 'z', 'p', 'q', 'r']
    axis_states_2 = ['vx', 'vy', 'vz', 'wp', 'wq', 'wr', 'ax', 'ay', 'az', 'ap', 'aq', 'ar']
    loc = 'upper right'
    #sig_type = 'círculo con físicas básicas'
    #sig_type = 'círculo con arrastre'
    fig, axs = plt.subplots(6, 3, figsize=(18,15), sharex=True)
    fig.suptitle(f'Trayectoria de muestra 1')
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
    
    
    fig.legend(ctrls,loc=10)
    plt.savefig('prueba.svg')
    plt.show()
    
    #
    input("Press Enter to exit...")