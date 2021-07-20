import time
from datetime import datetime
import numpy as np
import trajectories
import utils 

import sys
sys.path.append("./gym-pybullet-drones")
import pybullet
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
import gym
gym.logger.set_level(40)

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
            "uwp":35, "uwq":36, "uwr":37, 't':38}

POS_DICT = {'x':0, 'y':1, 'z':2, 'ux':0, 'uy':1, 'uz':2}

class Drone(object):
    def __init__(self, INIT_XYZS, INIT_RPYS,
                 control_timestep, control='',
                 i=0,
                 drone=DroneModel("cf2x")):
        self.drone=drone
        self.ctrl=control
        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.name = i
        self.control_timestep = control_timestep
        self.action = np.array([0,0,0,0])

    def setControl(self, control=None, drone_model=None):
        #### Initialize the controller ############################
        try:
            if control is None:
                if drone_model is not None:
                    if self.drone in [DroneModel.CF2X, DroneModel.CF2P]:
                        self.ctrl = DSLPIDControl(drone_model) 
                    elif self.drone in [DroneModel.HB]:
                        self.ctrl = SimplePIDControl(drone_model)
                else:
                    raise ValueError('controller or env is not setted')
            else: 
                self.ctrl=control
        except ValueError:
             print("Oops! Please enter the address of the controller and PIDparams or environment")

    def setPIDParams(self, PIDparams=[[]], PIDaxis=[]):
        pidDict = {'x':0, 'y':1, 'z':2, 'roll':3, 'pitch':4, 'yaw':5} 
        PIDconst = self.ctrl.getPIDconstXYZRPY()
        if len(PIDaxis)>0:
                for i, ax in enumerate(PIDaxis):
                    PIDconst[pidDict[ax]] = PIDparams[i]
        self.ctrl.setPIDconstXYZRPY(PIDconst[0], PIDconst[1], PIDconst[2],
                                    PIDconst[3], PIDconst[4], PIDconst[5])
        return PIDconst

    def computeControl(self, target_pos, target_vel, target_rpy, target_rpy_rates):
        if self.ctrl=='':
            print("Oops! Please initialize the controller first")
        else:
            self.action, _ , _ = self.ctrl.computeControlFromState(control_timestep=self.control_timestep,
                                                        state=self.state,
                                                        target_pos=self.INIT_XYZS + target_pos,
                                                        target_vel=target_vel,
                                                        target_rpy=self.INIT_RPYS + target_rpy,
                                                        target_rpy_rates=target_rpy_rates)
            self.ref = np.hstack([self.INIT_XYZS+target_pos, target_vel, self.INIT_RPYS+target_rpy, target_rpy_rates])
            
    def observe(self, state):
        self.state=state
    
    def __del__(self):
        attr = []
        for key in self.__dict__.keys():
            attr.append(key)
        for key in attr:
            delattr(self, key)

class PybulletSimDrone(object):
    def __init__(self, drone_type='cf2x',
                num_drones = 1,
                physics="pyb", 
                vision=False,
                gui=False,
                record_video=False,
                plot=False, 
                save_figure=False, 
                console_out=True,
                user_debug_gui=False,
                aggregate=False, 
                obstacles=False,
                save_data = False,
                data_path = '',
                dist_params = {},
                simulation_freq_hz=240,
                control_freq_hz=48,
                duration_sec=10):
                
        self.drone_type=drone_type
        self.num_drones=num_drones
        self.physics = physics
        self.vision = vision
        self.gui = gui
        self.record_video = record_video
        self.plot = plot
        self.save_figure = save_figure
        self.console_out=console_out
        self.user_debug_gui = user_debug_gui
        self.aggregate = aggregate
        self.obstacles = obstacles
        self.save_data = save_data
        self.data_path = data_path
        self.simulation_freq_hz = simulation_freq_hz
        self.control_freq_hz = control_freq_hz
        self.duration_sec = duration_sec
        self.disturbances = bool(dist_params)
        self.dist_params = dist_params
        self.AGGR_PHY_STEPS = int(self.simulation_freq_hz/self.control_freq_hz) if self.aggregate else 1
        self.INIT_XYZS = np.array([[i*0.2, i*0.2, 10+i*0.5] for i in range(self.num_drones)])
        self.INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/self.num_drones] for i in range(self.num_drones)])
        self.initializeEnv()

    def setdrones(self, drones):
        self.drones=drones
        self.num_drones=len(drones)
        self.INIT_XYZS = np.array([drone.INIT_XYZS for drone in self.drones])
        self.INIT_RPYS = np.array([drone.INIT_RPYS for drone in self.drones])
        self.initializeEnv()
        for drone in self.drones:
            if drone.ctrl=='':
                drone.setControl(drone_model=DroneModel(self.drone_type))

    def initializeEnv(self):
        if  hasattr(self, 'env'):
            self.env.close()
            self.gui_=self.gui
        else:
            self.gui_=False
        #### Create the environment with or without video capture ##
        self.env = CtrlAviary(drone_model=DroneModel(self.drone_type),
                            num_drones=self.num_drones,
                            initial_xyzs=self.INIT_XYZS,
                            initial_rpys=self.INIT_RPYS,
                            physics=Physics(self.physics),
                            neighbourhood_radius=10,
                            freq=self.simulation_freq_hz,
                            aggregate_phy_steps=self.AGGR_PHY_STEPS,
                            gui=self.gui_,
                            record=self.record_video,
                            obstacles=self.obstacles,
                            user_debug_gui=self.user_debug_gui
                            )

        #### Obtain the PyBullet Client ID from the environment ####
        self.PYB_CLIENT = self.env.getPyBulletClient()
        
        #### Initialize the logger #################################
        self.logger = Logger(logging_freq_hz=int(self.simulation_freq_hz/self.AGGR_PHY_STEPS),
                            num_drones=self.num_drones)

        self.CTRL_EVERY_N_STEPS = int(np.floor(self.env.SIM_FREQ/self.control_freq_hz))
        self.control_timestep = self.CTRL_EVERY_N_STEPS * self.env.TIMESTEP

        ### Disturbances variables ######
        if self.disturbances:
            self.l=0
            self.Rand_3 = [0.5]*len(self.dist_params['DIST_STATES'])
            self.DIST_EVERY_N_STEPS = np.ceil(self.duration_sec*self.env.SIM_FREQ/self.dist_params['N_DIST'])

    def initTrajectory(self, trajectories=[[]], axis=[], params=[{}]):
        self.NUM_WP = self.control_freq_hz*self.duration_sec
        self.wp_counters = np.array([0]*self.num_drones)
        ctrl_ax = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'r', 'wr']
        traj = {}
        for ax in ctrl_ax:
            traj[ax] = np.zeros(self.NUM_WP)
        for i, ax in enumerate(axis):
            traj[ax] = self.selTrajectory(type=trajectories[i], params=params[i])
        
        self.TARGET_POS = np.zeros((self.NUM_WP,3))
        self.TARGET_VEL = np.zeros((self.NUM_WP,3))
        self.TARGET_RPY = np.zeros((self.NUM_WP,3))
        self.TARGET_RPY_RATES = np.zeros((self.NUM_WP,3))
    
        for i in range(self.NUM_WP):
            self.TARGET_POS[i, :] = traj['x'][i], traj['y'][i], traj['z'][i]
            self.TARGET_VEL[i, :] = traj['vx'][i], traj['vy'][i], traj['vz'][i]
            self.TARGET_RPY[i, :] = 0, 0, traj['r'][i]
            self.TARGET_RPY_RATES[i, :] = 0, 0, traj['wr'][i]
    
    def step(self, action):
        self.obs, self.reward, self.done, self.info = self.env.step(action)
        for drone in self.drones:
        ######### Stability ########################################
            if (   (abs(self.obs[str(drone.name)]["state"][STATES_DICT['p']])>0.7)
                or (abs(self.obs[str(drone.name)]["state"][STATES_DICT['q']])>0.7)
                ):
                self.unstable = True
                print(f'***************Drone_{drone.name} UNSTABLE************')
                pass
    
    def disturb(self):
        self.l += 1
        for j in range(self.num_drones):
            ##### Disturbance ###################################
            for nd, d in enumerate(self.dist_params['DIST_STATES']):
                self.obs[str(j)]["state"][STATES_DICT[d]] += self.dist_params['D_FACTOR'][nd]*2*(self.Rand_3[nd]-0.5)
            
            if ("p" in self.dist_params['DIST_STATES']) or ("q" in self.dist_params['DIST_STATES']) or ("r" in self.dist_params['DIST_STATES']):
                quat = utils.euler_to_quaternion(self.obs[str(j)]["state"][7], #roll
                                                self.obs[str(j)]["state"][8], #pitch
                                                self.obs[str(j)]["state"][9])#yaw
                for q in range(len(quat)):
                    self.obs[str(j)]["state"][q+3] = quat[q]
        dt = np.random.rand()
        if self.l>(self.dist_params['DIST_TIME']*(1+dt))*self.CTRL_EVERY_N_STEPS+1:
            self.l=0
            self.Rand_3 = [0.5]*len(self.dist_params['DIST_STATES'])
            for i, p in enumerate(self.dist_params['D_PROB']):
                if np.random.rand()< p:
                    self.Rand_3[i] = np.random.rand()
    
    def computeControl(self, drone):
        #### Replay control for the current way point #############
        j = drone.name
        drone.computeControl(target_pos=self.TARGET_POS[self.wp_counters[j], :],
                            target_vel = self.TARGET_VEL[self.wp_counters[j], :],
                            target_rpy = self.TARGET_RPY[self.wp_counters[j], :], 
                            target_rpy_rates = self.TARGET_RPY_RATES[self.wp_counters[j], :]
                            )

        #### Go to the next way point and loop #####################
        self.wp_counters[j] = self.wp_counters[j] + 1 if self.wp_counters[j] < (self.NUM_WP-1) else 0

        return drone.action
    
    def log(self, i, drone):
        self.logger.log(drone=drone.name,
                        timestamp=i/self.env.SIM_FREQ,
                        state= drone.state,
                        control=drone.ref
                        )

    def runSim(self):
        #try:
            #### Run the simulation ####################################
            actions = {str(drone.name): np.array([0,0,0,0]) for drone in self.drones}
            START = time.time()
            self.unstable = False
            
            for i in range(0, int(self.duration_sec*self.env.SIM_FREQ), self.AGGR_PHY_STEPS):
                self.Rand_4 = np.random.rand()
                self.step(actions)
                
                if self.disturbances and ((i%self.DIST_EVERY_N_STEPS == 0) or self.l>0):
                    self.disturb()
                for drone in self.drones:
                    drone.observe(self.obs[str(drone.name)]["state"])
                    if i%self.CTRL_EVERY_N_STEPS == 0:
                        actions[str(drone.name)] = self.computeControl(drone)
                    self.log(i, drone)

            #### Printout ##############################################
                if i%self.env.SIM_FREQ == 0 and self.console_out:
                    self.env.render()

            #### Sync the simulation ###################################
                if self.gui:
                    sync(i, START, self.env.TIMESTEP)
                
                ######### Stability ########################################
                if self.unstable:
                    break

            #### Close the environment #################################
            self.env.close()

            #### Save the simulation results ###########################
            if self.save_data and not self.unstable:
                if self.data_path:
                    self.logger.save_csv(self.data_path)
                else:
                    self.logger.save_csv(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
            if (self.plot or self.save_figure) and not self.unstable:
                    self.logger.plot(save_figure=self.save_figure, plot=self.plot, name=self.data_path)
        #except:
        #    pass

            return self.logger
    
    def __del__(self):
        attr = []
        for key in self.__dict__.keys():
            attr.append(key)
        for key in attr:
            delattr(self, key)

    def selTrajectory(self, type, params):
        z = []
        if type=='step':
            z = trajectories.step(duration=self.NUM_WP,
                                  val=params['val'])
        elif type=='pulse':
            z = trajectories.pulse(duration=self.NUM_WP,
                                    val=params['val'],
                                    finish=params['finish'])
        elif type=='ramp':
            z = trajectories.ramp(duration=self.NUM_WP,
                                  m=params['val'],
                                  fs=self.control_freq_hz)  
        elif type=='square':
            z = trajectories.square(duration=self.NUM_WP,
                                  m=params['val'],
                                  fs=self.control_freq_hz) 
        elif type=='sin':
            z = trajectories.sin(duration=self.NUM_WP,
                                  m=params['val'],
                                  f=params['f'],
                                  fs=self.control_freq_hz)
        elif type=='noise':
            z = trajectories.noise(duration=self.NUM_WP,
                                  k=params['val'],
                                  f0=params['f0'],
                                  Order=params['order'],
                                  fs=self.control_freq_hz)
        elif type=='sawtooth_sweep':
            z = trajectories.sawtooth_sweep(duration=self.NUM_WP,
                                  k=params['val'],
                                  f0=params['f0'],
                                  f1=params['f1'],
                                  fs=self.control_freq_hz)
        elif type=='triangular_sweep':
            z = trajectories.triangular_sweep(duration=self.NUM_WP,
                                  k=params['val'],
                                  f0=params['f0'],
                                  f1=params['f1'],
                                  fs=self.control_freq_hz)
        elif type=='chirp':
            z = trajectories.chirp(duration=self.NUM_WP,
                                  k=params['val'],
                                  f0=params['f0'],
                                  f1=params['f1'],
                                  t1=params['t1'],
                                  method=params['method'],
                                  fs=self.control_freq_hz)
        elif type=='big_step_ret0':
            z = trajectories.big_step_ret0(length=self.NUM_WP,
                                  max=params['max'],
                                  min=params['min'],
                                  N_cycles=params['N_cycles'],
                                  each=params['each'])
        elif type=='big_step_notret0':
            z = trajectories.big_step_notret0(length=self.NUM_WP,
                                  max=params['max'],
                                  min=params['min'],
                                  N_cycles=params['N_cycles'],
                                  each=params['each'])                          
        elif type=='step_notret0':
            z = trajectories.step_notret0(length=self.NUM_WP,
                                  max=params['max'],
                                  min=params['min'],
                                  N_cycles=params['N_cycles'],
                                  each=params['each'])
        elif type=='ramp_step_notret0':
            z = trajectories.ramp_step_notret0(length=self.NUM_WP,
                                  max=params['max'],
                                  min=params['min'],
                                  N_cycles=params['N_cycles'],
                                  each=params['each'])
        elif type=='step_ret0':
            z = trajectories.step_ret0(length=self.NUM_WP,
                                  max=params['max'],
                                  min=params['min'],
                                  N_cycles=params['N_cycles'],
                                  each=params['each'])
        elif type=='random_step':
            z = trajectories.random_step(length=self.NUM_WP,
                                  K=params['val'],
                                  DC=params['DC'],
                                  each=params['each'])  
        elif type=='stopped':
            z = trajectories.stopped(self.NUM_WP, DC=params['val'])
        else:
            z = trajectories.stopped(self.NUM_WP, 0)
        return z
