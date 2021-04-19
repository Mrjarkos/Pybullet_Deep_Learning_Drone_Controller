import os
import time
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'
STATES_DICT = {"x":0, "y":1, "z":2,
            "Q1":3, "Q2":4, "Q3":5, "Q4":6,
            "p":7, "q":8, "r":9,
            "vx":10, "vy":11, "vz":12,
            "wp":13, "wq":14, "wr":15,
            "ax":16, "ay":17, "az":18,
            "ap":19, "aq":20, "ar":21,
            "RPM0":22, "RPM1":23, "RPM2":24, "RPM3":25,
            "ux":26, "uy":27, "uz":28,
            "uvx":29, "uvy":30, "uvz":31,
            "up":32, "uq":33, "ur":34,
            "uwp":35, "uwq":36, "uwr":37}

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 num_drones: int=1,
                 duration_sec: int=0
                 ):
        """Logger class __init__ method.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #self.states = np.zeros((num_drones, 16, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x, pos_y, pos_z,
        self.states = np.zeros((num_drones, 26, duration_sec*self.LOGGING_FREQ_HZ)) #### 26 states: pos_x, pos_y, pos_z,
                                                                                                  # Q1, Q2, Q3, Q4
                                                                                                  # roll, pitch, yaw,
                                                                                                  # vel_x, vel_y, vel_z,
                                                                                                  # ang_vel_x, ang_vel_y, ang_vel_z,
                                                                                                  # acc_x, acc_y, acc_z,
                                                                                                  # ang_acc_x, ang_acc_y, ang_acc_z,
                                                                                                  # rpm0, rpm1, rpm2, rpm3
        #### Note: this is not the same order nor length ###########
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                             # pos_y,
                                                                                                             # pos_z,
                                                                                                             # vel_x, 
                                                                                                             # vel_y,
                                                                                                             # vel_z,
                                                                                                             # roll,
                                                                                                             # pitch,
                                                                                                             # yaw,
                                                                                                             # ang_vel_x,
                                                                                                             # ang_vel_y,
                                                                                                             # ang_vel_z

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 26 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            #self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 26, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.states[drone, :, current_counter] = state
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    def getStates(self, drone, states):
        st_dict = {}
        for state in states:
            if state[0]=='u':
                st_dict[state] = self.controls[drone, STATES_DICT[state]-26, :]
            else:
                st_dict[state] = self.states[drone, STATES_DICT[state], :]
        return st_dict
    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(os.path.dirname(os.path.abspath(__file__))+"/../../files/logs/save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy", 'wb') as out_file:
            np.save(out_file, self.timestamps)
            np.save(out_file, self.states)
            np.save(out_file, self.controls)
    
       ################################################################################

    def save_csv(self, name="", path=""):
        """Save the logs to file.
        """
        timestamp = pd.DataFrame(data=self.timestamps.T, columns=["timestamps"]) 
        states = pd.DataFrame(data=self.states[0].T, columns=["x", "y", "z",
                                                            "Q1", "Q2", "Q3", "Q4",
                                                            "p", "q", "r",
                                                            "vx", "vy", "vz",
                                                            "wp", "wq", "wr",
                                                            "ax", "ay", "az",
                                                            "ap", "aq", "ar",
                                                            "RPM0", "RPM1", "RPM2", "RPM3"])
        controls = pd.DataFrame(data = self.controls[0].T, columns=["ux", "uy", "uz",
                                                            "uvx", "uvy", "uvz",
                                                            "up", "uq", "ur",
                                                            "uwp", "uwq", "uwr"])
        result = pd.concat([timestamp, states], axis=1, join='inner')

        if name == "":
            name = "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        if path == "":
            path = os.path.dirname(os.path.abspath(__file__))+"/../../files/logs/"
        pd.concat([result, controls], axis=1, join='inner').to_csv(path+name+".csv", index=False)
    
    def plot(self, pwm=False, save_figure=False, name="", path="", format='png'):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'g']) + cycler('linestyle', ['-', '--', '-', '--'])))
        fig, axs = plt.subplots(10, 2, figsize=(15,15))
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 0, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 1, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 2, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 6, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 7, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 8, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 9, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 10, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 11, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        #### Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 3, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 4, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
            axs[row, col].plot(t, self.controls[j, 5, :], label="ref_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        #### XYZ Acc #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 16, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ax (m/s^2)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 17, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ay (m/s^2)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 18, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('az (m/s^2)')

        #### RPY Rates #############################################

        # row = 3
        # for j in range(self.NUM_DRONES):
        #     rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, rdot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('rdot (rad/s)')
        # row = 4
        # for j in range(self.NUM_DRONES):
        #     pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, pdot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('pdot (rad/s)')
        # row = 5
        # for j in range(self.NUM_DRONES):
        #     ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
        #     axs[row, col].plot(t, ydot, label="drone_"+str(j))
        # axs[row, col].set_xlabel('time')
        # axs[row, col].set_ylabel('ydot (rad/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(22,26):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 22, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 23, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 24, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 25, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        plt.show(block=False)

        if save_figure:
            if name == "":
                name = "save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
            if path == "":
                path = os.path.dirname(os.path.abspath(__file__))+"/../../files/logs/"
            plt.savefig(path+name+'.'+format, dpi=300, format=format) 
