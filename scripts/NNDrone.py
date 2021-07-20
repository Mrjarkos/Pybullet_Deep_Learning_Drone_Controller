from re import S
import tensorflow as tf
assert (tf.__version__=='2.5.0'), 'Versión incorrecta de Tensorflow, por favor instale 2.5.0'
import numpy as np
assert (np.__version__=='1.19.5'), 'Versión incorrecta de Numpy, por favor instale 1.19.5'
import pandas as pd
import collections

gpus = tf.config.list_physical_devices('GPU') 
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import sys
sys.path.append("./gym-pybullet-drones")

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from DronePySim import STATES_DICT, POS_DICT

class NNDrone(object):
    def __init__(self, INIT_XYZS, INIT_RPYS, i=0):
        self.ctrl=''
        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.name = i
        self.action = np.array([0,0,0,0])

    def initControl(self,
                    model_path,
                    norm_data_path,
                    input_list,
                    output_list,
                    window):
        self.ctrl = tf.keras.models.load_model(model_path)
        self.norm_data = pd.read_csv(norm_data_path)
        self.input_list = input_list
        self.output_list = output_list
        self.window = window
        self.states = collections.deque(maxlen=self.window)
        self.ref_control = [0]*12
        
    def computeControl(self, target_pos, target_vel, target_rpy, target_rpy_rates):
        if self.ctrl=='':
            print("Oops! Please initialize the controller first")
        else:
            #### Update Last control
            target_pos_=self.INIT_XYZS + target_pos
            target_rpy_=self.INIT_RPYS + target_rpy
            self.ref_control = target_pos_.tolist() + target_vel.tolist() + target_rpy_.tolist() + target_rpy_rates.tolist()
            self.states.pop()
            self.observe(self.state)
            nn_input = np.array(self.states)
            if(len(nn_input)<self.window):
                nn_input = nn_input.reshape(1, len(nn_input), len(self.input_list))
            else:
                nn_input = nn_input.reshape(1, self.window, len(self.input_list))
            pred = self.ctrl.predict(nn_input)
            self.action = self.Norm_inv(pred)
            self.ref = np.hstack([self.INIT_XYZS+target_pos, target_vel, self.INIT_RPYS+target_rpy, target_rpy_rates])

    def observe(self, state):
        self.state=list(state)
        current_state = self.state+self.ref_control
        current_state = [current_state[STATES_DICT[x]] for x in self.input_list]
        current_state = self.Norm(current_state)
        self.states.append(current_state) 
    
    def Norm(self, state):
        df_norm = []
        for i, prop in enumerate(self.input_list):
            # 1 ~ Mean  7 ~ Max  3 ~ Min
            if prop in POS_DICT.keys():
                H = self.norm_data[prop][1]-self.INIT_XYZS[POS_DICT[prop]]                
                x_temp = (state[i]-self.norm_data[prop][1]+H)/(
                            self.norm_data[prop][7]-self.norm_data[prop][3])
            else:                    
                x_temp = (state[i]-self.norm_data[prop][1])/(
                            self.norm_data[prop][7]-self.norm_data[prop][3])
            df_norm.append(x_temp)
        return np.array(df_norm)

    def Norm_inv(self, output):
        rpm=output[0]
        df_norm = []
        for i, prop in enumerate(self.output_list):
                # 1 ~ Mean  7 ~ Max  3 ~ Min
                x_temp = rpm[i]*(self.norm_data[prop][7]-self.norm_data[prop][3])+self.norm_data[prop][1]
                df_norm.append(x_temp)
        return np.array(df_norm).reshape(output.shape)
    
    def __del__(self):
        attr = []
        for key in self.__dict__.keys():
            attr.append(key)
        for key in attr:
            delattr(self, key)
        