import argparse
import math
from click import style
import numpy as np
import sys

from trajectories import *
# import sys
# sys.path.append('../')
from DronePySim import Drone, PybulletSimDrone 
from NNDrone import *
from trajectories import *
from sys_info import *
import json
from gym_pybullet_drones.utils.utils import str2bool

import itertools
import operator
import os
  
step_time = 8

COLOR_CTRLS={
            'DSLPID':'tab:blue', 
            'ANN': 'tab:red',
            'ANN_feedback':'tab:purple',
            'LSTM':'tab:olive',
            #'LSTMCNN':'tab:green',
            'LSTMCNN':'tab:red',
            'CLSTM':'tab:cyan',
             #'Referencia':'tab:orange'
             'Referencia':'k'
             }

COLOR_CTRLS={
            'DSLPID':'tab:blue', 
            # 'LSTMCNN_8': 'tab:red',
            # 'LSTMCNN_16':'tab:purple',
            # 'LSTMCNN_32':'tab:olive',
            # 'LSTMCNN_64':'tab:green',
            'LSTMCNN':'tab:red',
            # 'CLSTM':'tab:cyan',
             #'Referencia':'tab:orange'
             'Referencia':'k'
             }


model_path = {
        #'ANN': '../Models/ANN/Dataset_Final_ANN_9.h5',
        #'LSTM':'../Models/LSTM/Dataset_Final_LSTM_3.h5',
        'LSTMCNN':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_4':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_8':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_16':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_32':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_64':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        # 'LSTMCNN_128':'../Models/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5',
        #'CLSTM':'../Models/CLSTM/Dataset_Final_CLSTM_Tuner_2_2.h5',
    }


# COLOR_CTRLS={
#             'DSLPID':'tab:blue', 
#             'ANN_1': 'tab:red',
#             'ANN_2':'tab:purple',
#             'ANN_3':'tab:olive',
#             #'ANN_3':'tab:red',
#             'ANN_4':'tab:green',
#             'ANN_5':'tab:cyan',
#              'Referencia':'k'
#              }

# model_path = {
#     'ANN_1':'../tmp/ANN/Dataset_Final_ANN_0.h5', #1
#     'ANN_2':'../tmp/ANN/Dataset_Final_ANN_5.h5', #2
#     'ANN_3':'../tmp/ANN/Dataset_Final_ANN_9.h5', #3
#     'ANN_4':'../tmp/ANN/Dataset_Final_ANN_8.h5', #4
#     'ANN_5':'../tmp/ANN/Dataset_Final_ANN_10.h5' #5
#     }

# COLOR_CTRLS={
#             'DSLPID':'tab:blue', 
#             'ANN_feedback_1': 'tab:red',
#             'ANN_feedback_2':'tab:purple',
#             'ANN_feedback_3':'tab:olive',
#             'ANN_feedback_4':'tab:green',
#             'ANN_feedback_5':'tab:cyan',
#              'Referencia':'k'
#              }

# model_path = {
#     'ANN_feedback_1':'../tmp/ANN_Feedback/Dataset_Final_ANN_Feedback_0.h5', #1
#     'ANN_feedback_2':'../tmp/ANN_Feedback/Dataset_Final_ANN_Feedback_3.h5', #2
#     'ANN_feedback_3':'../tmp/ANN_Feedback/Dataset_Final_ANN_Feedback_7.h5', #3
#     'ANN_feedback_4':'../tmp/ANN_Feedback/Dataset_Final_ANN_Feedback_8.h5', #4
#     'ANN_feedback_5':'../tmp/ANN_Feedback/Dataset_Final_ANN_Feedback_9.h5', #5
#     }

# COLOR_CTRLS={
#             'DSLPID':'tab:blue', 
#             'LSTM_1': 'tab:red',
#             'LSTM_2':'tab:purple',
#             'LSTM_3':'tab:olive',
#             'LSTM_4':'tab:green',
#             'LSTM_5':'tab:cyan',
#              'Referencia':'k'
#              }

# model_path = {
#     'LSTM_1':'../tmp/LSTM/Dataset_Final_LSTM_0.h5', #1
#     'LSTM_2':'../tmp/LSTM/Dataset_Final_LSTM_1_2.h5', #2
#     'LSTM_3':'../tmp/LSTM/Dataset_Final_LSTM_2_2.h5', #3
#     'LSTM_4':'../tmp/LSTM/Dataset_Final_LSTM_3.h5', #4
#     'LSTM_5':'../tmp/LSTM/Dataset_Final_LSTM_4.h5', #5
#     }

# model_path = {
#     'LSTMCNN_1':'../tmp/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_0.h5', #1
#     'LSTMCNN_2':'../tmp/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_1.h5', #2
#     'LSTMCNN_3':'../tmp/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_2.h5', #3
#     'LSTMCNN_4':'../tmp/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_3.h5', #4
#     'LSTMCNN_5':'../tmp/LSTMCNN/Dataset_Final_LSTMCNN_Tuner_4.h5' #5
#     }

# COLOR_CTRLS={
#             'DSLPID':'tab:blue', 
#             'LSTMCNN_1': 'tab:red',
#             'LSTMCNN_2':'tab:purple',
#             'LSTMCNN_3':'tab:olive',
#             'LSTMCNN_4':'tab:green',
#             'LSTMCNN_5':'tab:cyan',
#              'Referencia':'k'
#              }

# model_path = {
#     'CLSTM_1':'../tmp/CLSTM/Dataset_Final_CLSTM_0.h5', #1
#     'CLSTM_2':'../tmp/CLSTM/Dataset_Final_CLSTM_Tuner_0_2.h5', #2
#     'CLSTM_3':'../tmp/CLSTM/Dataset_Final_CLSTM_Tuner_3.h5', #3
#     'CLSTM_4':'../tmp/CLSTM/Dataset_Final_CLSTM_Tuner_2_2.h5', #4
#     'CLSTM_5':'../tmp/CLSTM/Dataset_Final_CLSTM_Tuner_1_2.h5' #5
#     }

# COLOR_CTRLS={
#             'DSLPID':'tab:blue', 
#             'CLSTM_1': 'tab:red',
#             'CLSTM_2':'tab:purple',
#             'CLSTM_3':'tab:olive',
#             'CLSTM_4':'tab:green',
#             'CLSTM_5':'tab:cyan',
#              'Referencia':'k'
#              }

window = 64
root = '../logs/Datasets'
dataset =  f'Dataset_Final'
norm_data_path = f"{root}/data_description_{dataset}.csv"
        
control_states = ['x', 'y', 'z', 'r']
states_list = ['x', 'y','z','p','q','r','vx','vy','vz',
                'wp','wq','wr','ax','ay','az','ap','aq',
                'ar','ux','uy','uz', 'ur']
rpm_list = ['RPM0', 'RPM1', 'RPM2', 'RPM3']
RPM_B = 14468.429188505048
if __name__ == "__main__":
    
     #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb_gnd",      type=str,       help='Physics updates (default: PYB)', metavar='')
    parser.add_argument('--gui',                default=False,      type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=False,      type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=30,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--save_data',          default=False,       type=str2bool,      help='Whether to save the data (default: True)', metavar='')
    ARGS = parser.parse_args()

    H=1
    axis = ['z']
    #axis = ['z', 'x', 'y', 'r']
    #trajectories=['step']#, 'sin', 'cos', 'stop']
    #params=[{'val':1}]
    #         {'val':0.3, 'f':0.5},
    #         {'val':0.3, 'f':0.5},
    #         {'val':0.0}]
    trajectories=['step']#, 'step', 'step', 'step']
    params=[{'val':0.0}]
            # {'val':0.1},
            # {'val':0.1},
            # {'val':0.1}]
    
    # axis=['x', 'y','z']
    # f = 0.4
    # d= 9
    # params=[]
    # trajectories =lemniscate(ARGS.duration_sec*ARGS.control_freq_hz, d, f, ARGS.control_freq_hz, True)
    # trajectories.append(ramp(ARGS.duration_sec*ARGS.control_freq_hz, 0.15, ARGS.control_freq_hz))
    #trajectories.append(stopped(ARGS.duration_sec*ARGS.control_freq_hz, 0.0))
    
#    NN_type = 'ANN'
    NN_type = 'LSTMCNN'
    base_controller = 'DSLPID'
    ref_label = 'Referencia'
    #ctrls = [base_controller, NN_type, ref_label]
    ctrls = [base_controller]+[*model_path]+[ref_label]
    #ctrls = [base_controller, 'ANN', 'LSTM', 'LSTMCNN', 'CLSTM', ref_label]
    output={}
    
    trayectoria = 'quieto'
    dir = f'../Imágenes/DSLPID VS NN/'
    try:
    # Create target Directory
        os.mkdir(dir+trayectoria)
        print("Directory " , trayectoria ,  " Created ") 
    except FileExistsError:
        print("Directory " , trayectoria ,  " already exists")
        
    try:
    # Create target Directory
        os.mkdir(dir+trayectoria+'/'+NN_type)
        print("Directory " , NN_type ,  " Created ") 
    except FileExistsError:
        print("Directory " , NN_type ,  " already exists")
    
    if len(ctrls)==3:
        name = f'{ctrls[0]}_vs_{ctrls[1]}_{trayectoria}'
        title = f'{ctrls[0]} vs {ctrls[1]} con {trayectoria}'
    else:
        name = f'Comparación_controladores_con_{trayectoria}'
        title = f'Comparación controladores con {trayectoria}'
        
    title_size = 16
    rect=[0.05, 0.05, 0.85, 0.95]
    alpha=0.7
    beta=0.85
    
    DIST_STATES = ['vz', 'p', 'q', 'r']
    D_FACTOR = 3*[1, -0.2, 0.2, 0.25]
    #D_PROB = [0, 1, 1, 0] X,Y
    D_PROB = [0, 0, 0, 0] #Z
    #D_PROB = [1, 1, 1, 1] #r
    DIST_TIME = 6
    N_DIST = 7#ARGS.duration_sec//2
    dist_params = {
        'DIST_STATES' : DIST_STATES,
        'D_FACTOR' : D_FACTOR,
        'D_PROB': D_PROB,
        'DIST_TIME' : DIST_TIME,
        'N_DIST' : N_DIST
    }
    
    logger = []
    for i in range(len(ctrls)):
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
                                    duration_sec=ARGS.duration_sec,
                                    step_time = step_time
                                    )
        drones = []
        if ctrls[i]==base_controller:
            drones.append(
                    Drone(INIT_XYZS=[0,0,H],
                        INIT_RPYS=[0,0,0],
                        control_timestep=pySim.control_timestep,
                        i=0,
                        )
                    )
        elif ctrls[i]==ref_label:
            continue
        else:
            drones.append(
                    Load_NN_controller(i=0,
                        INIT_XYZS=[0,0,H],
                        INIT_RPYS=[0,0,0],
                        model_path=model_path[ctrls[i]],
                        norm_data_path=norm_data_path,
                        type_controller=ctrls[i],
                        input_list=states_list,
                        output_list=rpm_list,
                        window=window,
                        )
                    )

        pySim.setdrones(drones)
        pySim.initTrajectory(trajectories=trajectories,
                                params=params,
                                axis=axis)
        logger.append(pySim.runSim())
    #############################################################################################3    
    fig, axs = plt.subplots(6, 3, figsize=(18,15), sharex=True)
    #fig.suptitle(title, fontsize=title_size)
    
    print(f'******************Trayectoria = {trayectoria}*********')
    output["name"]=trayectoria
    output["ref"]={}
    
    # for i in range(len(axis)):
    #     print(f'{axis[i]}={trajectories[i]} {params[i]["val"]}')
    #     output["ref"][axis[i]]={}
    #     output["ref"][axis[i]]['trajectory']=trajectories[i]
    #     output["ref"][axis[i]]['params']=params[i]["val"]
        
    output['ctrls']={}
    
    for i in range(len(ctrls)):
        
        if ctrls[i]==ref_label:
            continue
        
        print(f'*********Controller = {ctrls[i]}*********')
        output['ctrls'][ctrls[i]]={}
        if ctrls[i] != base_controller:
            output['ctrls'][ctrls[i]]['path'] = model_path[ctrls[i]]
            
        output['ctrls'][ctrls[i]]['Stable']=logger[i].get_stability(0)
        log = logger[i].getStates(0)
        t = log['t']
        
        axs[0, 0].plot(t, log['x'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[0, 1].plot(t, log['vx'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[0, 2].plot(t, log['ax'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        axs[1, 0].plot(t, log['y'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[1, 1].plot(t, log['vy'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[1, 2].plot(t, log['ay'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        axs[2, 0].plot(t, log['z'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[2, 1].plot(t, log['vz'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[2, 2].plot(t, log['az'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        axs[3, 0].plot(t, log['r'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[3, 1].plot(t, log['wr'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[3, 2].plot(t, log['ar'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        axs[4, 0].plot(t, log['p'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[4, 1].plot(t, log['wp'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[4, 2].plot(t, log['ap'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        axs[5, 0].plot(t, log['q'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[5, 1].plot(t, log['wq'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[5, 2].plot(t, log['aq'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        
        for j in range(len(control_states)):
            print(f'*********State = {control_states[j]}*********')
            output['ctrls'][ctrls[i]][control_states[j]]={}
            yout= log[control_states[j]]
            u = log['u'+control_states[j]]
            
            y_ts, ts = settling_time(yout, t)
            tr, tr_i, yr_i, tr_f, yr_f = rise_time(yout, t, u[-1])
            os, tp, y_max = overshoot(yout, t, u[-1])
            iste = ISTE(t, yout, u)
            mse = MSE(yout, u)
            e_ss, e_ss_abs = ess(y_ts, u[-1])
            
            print(f'Ts = {ts-step_time}, Ys = {y_ts}')
            print(f'Os = {os}%, Tp = {tp-step_time}, y_max={y_max}')
            print(f'Tr = {tr}')
            print(f'Tr_i = {tr_i-step_time}, Tr_f = {tr_f-step_time}')
            print(f'e_ss_abs = {e_ss_abs}, e_ss ={e_ss}')
            print(f'iste = {iste}')
            print(f'mse = {mse}')
            
            output['ctrls'][ctrls[i]][control_states[j]]['Ts']=ts-step_time
            output['ctrls'][ctrls[i]][control_states[j]]['Ys']=y_ts
            output['ctrls'][ctrls[i]][control_states[j]]['Os']=os
            output['ctrls'][ctrls[i]][control_states[j]]['Tp']=tp-step_time
            output['ctrls'][ctrls[i]][control_states[j]]['y_max']=y_max
            output['ctrls'][ctrls[i]][control_states[j]]['Tr']=tr
            output['ctrls'][ctrls[i]][control_states[j]]['Tr_i']=tr_i-step_time
            output['ctrls'][ctrls[i]][control_states[j]]['Tr_f']=tr_f-step_time
            output['ctrls'][ctrls[i]][control_states[j]]['e_ss_abs']=e_ss_abs
            output['ctrls'][ctrls[i]][control_states[j]]['e_ss']=e_ss
            output['ctrls'][ctrls[i]][control_states[j]]['iste']=iste
            output['ctrls'][ctrls[i]][control_states[j]]['mse']=mse
        
        Q = 0
        for j in range(len(rpm_list)):
            Q+=MSE(np.absolute(log[rpm_list[j]]),RPM_B)
        Q/=len(rpm_list)
        output['ctrls'][ctrls[i]]['Q']=Q
        print(f'Q = {Q}')
        
    for i in range(len(ctrls)-1):
        Total = 0
        for j in range(len(control_states)):
            output['ctrls'][ctrls[i]][control_states[j]]['Ts_norm'] = output['ctrls'][ctrls[i]][control_states[j]]['Ts']/output['ctrls'][base_controller][control_states[j]]['Ts']
            output['ctrls'][ctrls[i]][control_states[j]]['Os_norm'] = output['ctrls'][ctrls[i]][control_states[j]]['Os']/output['ctrls'][base_controller][control_states[j]]['Os']
            output['ctrls'][ctrls[i]][control_states[j]]['iste_norm'] = output['ctrls'][ctrls[i]][control_states[j]]['iste']/output['ctrls'][base_controller][control_states[j]]['iste']
            output['ctrls'][ctrls[i]][control_states[j]]['Total_ax'] = (1/3)*(output['ctrls'][ctrls[i]][control_states[j]]['Ts_norm']+output['ctrls'][ctrls[i]][control_states[j]]['Os_norm']+output['ctrls'][ctrls[i]][control_states[j]]['iste_norm'])
            Total += output['ctrls'][ctrls[i]][control_states[j]]['Total_ax']
        Total/=len(control_states)
        output['ctrls'][ctrls[i]]['Total'] = Total
        
    axs[0, 0].plot(log['t'], log['ux'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--')
    axs[1, 0].plot(log['t'], log['uy'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta,  linestyle='--')
    axs[2, 0].plot(log['t'], log['uz'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta,  linestyle='--')
    axs[3, 0].plot(log['t'], log['ur'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta,  linestyle='--')
    
    axs[5, 0].set_xlabel('Tiempo (s)')
    axs[5, 1].set_xlabel('Tiempo (s)')
    axs[5, 2].set_xlabel('Tiempo (s)')
    
    axs[0, 0].set_ylabel('x (m)')
    axs[0, 1].set_ylabel(r'$v_x$ (m/s)')
    axs[0, 2].set_ylabel(r'$a_x$ ($m/s^2$)')
    axs[1, 0].set_ylabel('y (m)')
    axs[1, 1].set_ylabel(r'$v_y$ ($m/s$)')
    axs[1, 2].set_ylabel(r'$a_y$ ($m/s^2$)')
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
    
    for _, ax in enumerate(axs.flat):
        ax.grid()
    
    fig.legend(ctrls,loc=7)
    plt.tight_layout(rect=[0.05, 0.05, 0.90, 0.95])
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/states.svg')
    f = open(f"{dir}{trayectoria}/{NN_type}/values.json", "w")
    f.write(json.dumps(output))
    f.close()    
    #############################################################################################3
    fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex=True)
    #fig.suptitle(f'Error acumulado', fontsize=title_size)
    for j, ax in enumerate(axs.flat):
        if ctrls[i]==ref_label:
            continue
        
        for i in range(len(ctrls)-1):
            log = logger[i].getStates(0)
            yout= log[control_states[j]]
            u = log['u'+control_states[j]]
            e=[k for k in itertools.accumulate(abs(yout-u))]
            #print(f'[{j}][{i}] e_acumm={e[-1]}')
            ax.plot(t, e, label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        ax.set_xlabel('Tiempo (s)')
        if control_states[j] == 'r':
            ax.set_title(r'$\psi$ (rad)')
        else:
            ax.set_title(f'{control_states[j]} (m)')
        ax.grid()
    fig.legend(ctrls[:-1],loc=7)

    plt.tight_layout(rect=rect)    
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/error.svg')
    #############################################################################################3
    
    fig, axs = plt.subplots(2, 2, figsize=(8,8), sharex=True)
    #fig.suptitle(f'Actuadores', fontsize=title_size)
    for j, ax in enumerate(axs.flat):
        if ctrls[i]==ref_label:
            continue
        
        for i in range(len(ctrls)-1):
            log = logger[i].getStates(0)
            ax.plot(t, log[rpm_list[j]], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        ax.set_xlabel('Tiempo (s)')
        ax.set_title(f'{rpm_list[j]} (rpm)')
        ax.grid()
    fig.legend(ctrls[:-1],loc=7)
    plt.tight_layout(rect=rect)
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/output.svg')
    #############################################################################################
    
    fig = plt.figure(figsize=(8,8))
    #fig.suptitle(f'Vista 3D', fontsize=title_size)
    ax = plt.axes(projection='3d')
    for i in range(len(ctrls)-1):
        log = logger[i].getStates(0)
        ax.plot3D(log['x'], log['y'], log['z'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha);
    ax.plot3D(log['ux'], log['uy'], log['uz'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
 
    ax.grid()
    ax.legend(ctrls,loc=0)
    plt.tight_layout()
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/3D.svg')
    
    #############################################################################################
    fig, (ax1, ax2)= plt.subplots(1, 2, figsize=(8,4))
    #fig.suptitle(f'Vista planos', fontsize=title_size)
    for i in range(len(ctrls)-1):
        log = logger[i].getStates(0)
        ax1.plot(log['x'], log['y'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha);
        ax2.plot(log['t'], log['z'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha);
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_title('Plano xy')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('z (m)')
        ax2.set_title('Plano z')
    ax1.plot(log['ux'], log['uy'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');
    ax2.plot(log['t'], log['uz'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');
    ax1.grid()
    ax2.grid()
    fig.legend(ctrls,loc=7)
    plt.tight_layout(rect=rect)
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/planos.svg')
    #############################################################################################
    fig, axs= plt.subplots(2, 2, figsize=(8,8))
    #fig.suptitle(f'Vista planos', fontsize=title_size)
    for i in range(len(ctrls)-1):
        log = logger[i].getStates(0)
        for j, ax in enumerate(axs.flat):
            ax.plot(log['t'], log[control_states[j]], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
                

    for j, ax in enumerate(axs.flat):
        ax.plot(log['t'], log['u'+control_states[j]], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');
        ax.set_xlabel('Tiempo (s)')
        if control_states[j]=='r':
            ax.set_ylabel(r'$\psi$ (rad)')
        else:
            ax.set_ylabel(f'{control_states[j]} (m)')
        ax.grid()
    fig.legend(ctrls,loc=7)
    plt.tight_layout(rect=rect)
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/control_states.svg')
    ##########################################################################
    # fig = plt.figure(figsize=(8,8))
    # #fig.suptitle(f'Vista 3D', fontsize=title_size)
    # ax = plt.axes()
    # for i in range(len(ctrls)-1):
    #     log = logger[i].getStates(0)
    #     ax.plot(log['t'], log['z'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha);
    # ax.plot(log['t'], log['uz'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');
    # ax.set_xlabel('Tiempo (s)')
    # ax.set_ylabel(r'z (m)')
    # ax.grid()
    # ax.legend(ctrls,loc=0)
    # plt.tight_layout()
    # plt.savefig(f'{dir}{trayectoria}/{NN_type}/z.svg')
    
    ##########################################################################
    fig, axs= plt.subplots(3, 1, figsize=(8,8), sharex=True)
    #fig.suptitle(f'Vista planos', fontsize=title_size)
    for i in range(len(ctrls)-1):
        log = logger[i].getStates(0)
        axs[0].plot(log['t'], log['z'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[1].plot(log['t'], log['vz'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
        axs[2].plot(log['t'], log['az'], label=ctrls[i], color=COLOR_CTRLS[ctrls[i]], alpha=alpha)
    
    axs[0].plot(log['t'], log['uz'], label=ctrls[-1], color=COLOR_CTRLS[ctrls[-1]], alpha=alpha*beta, linestyle='--');    
    
    axs[0].set_xlabel('Tiempo (s)')
    axs[1].set_xlabel('Tiempo (s)')
    axs[2].set_xlabel('Tiempo (s)')
    
    axs[0].set_ylabel('z (m)')
    axs[1].set_ylabel(r'$v_z$ (m/s)')
    axs[2].set_ylabel(r'$a_z$ (m/s$^2$)')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    fig.legend(ctrls,loc=7)
    plt.tight_layout(rect=rect)
    plt.savefig(f'{dir}{trayectoria}/{NN_type}/z_states.svg')
    
    
    plt.show()    
    input("Press Enter to exit...")
    
    