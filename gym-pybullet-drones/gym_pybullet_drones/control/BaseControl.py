import math
import numpy as np
import pybullet as p
from enum import Enum
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.envs.BaseAviary import BaseAviary

class BaseControl(object):
    """Base class for control.

    Implements `__init__()`, `reset(), and interface `computeControlFromState()`,
    the main method `computeControl()` should be implemented by its subclasses.

    """

    ################################################################################

    def __init__(self,
                 env: BaseAviary
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        env : BaseAviary
            The simulation environment to control.

        """
        #### Set general use constants #############################
        self.DRONE_MODEL = env.DRONE_MODEL
        """int: The number of drones in the simulation environment."""
        self.GRAVITY = env.GRAVITY
        """float: The gravitational force (M*g) acting on each drone."""
        self.KF = env.KF
        """float: The coefficient converting RPMs into thrust."""
        self.KM = env.KM
        """float: The coefficient converting RPMs into torque."""
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control classes.

        A general use counter is set to zero.

        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3),
                                vel_ctrl = False
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        return self.computeControl(vel_ctrl = vel_ctrl,
                                   control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy,
                                   target_vel=target_vel,
                                   target_rpy_rates=target_rpy_rates
                                   )
    def setPIDconst(self,
                    P_COEFF_FOR, I_COEFF_FOR, D_COEFF_FOR,
                    P_COEFF_TOR, I_COEFF_TOR, D_COEFF_TOR):
        if P_COEFF_FOR is not None:
            self.P_COEFF_FOR = P_COEFF_FOR
        if I_COEFF_FOR is not None:
            self.I_COEFF_FOR = I_COEFF_FOR
        if D_COEFF_FOR is not None:
            self.D_COEFF_FOR = D_COEFF_FOR
        if P_COEFF_TOR is not None:
            self.P_COEFF_TOR = P_COEFF_TOR
        if I_COEFF_TOR is not None:
            self.I_COEFF_TOR = I_COEFF_TOR
        if D_COEFF_TOR is not None:
            self.D_COEFF_TOR = D_COEFF_TOR

    def setPIDconstXYZRPY(self, PID_X, PID_Y, PID_Z,
                          PID_ROLL, PID_PITCH, PID_YAW):
        self.P_COEFF_FOR = np.array([PID_X[0],PID_Y[0], PID_Z[0]])
        self.I_COEFF_FOR = np.array([PID_X[1],PID_Y[1], PID_Z[1]])
        self.D_COEFF_FOR = np.array([PID_X[2],PID_Y[2], PID_Z[2]])
        self.P_COEFF_TOR = np.array([PID_ROLL[0],PID_PITCH[0], PID_YAW[0]])
        self.I_COEFF_TOR = np.array([PID_ROLL[1],PID_PITCH[1], PID_YAW[1]])
        self.D_COEFF_TOR = np.array([PID_ROLL[2],PID_PITCH[2], PID_YAW[2]])

    def getPIDconst(self):
        return [self.P_COEFF_FOR, 
                self.I_COEFF_FOR, 
                self.D_COEFF_FOR, 
                self.P_COEFF_TOR, 
                self.I_COEFF_TOR, 
                self.D_COEFF_TOR]

    def getPIDconstXYZRPY(self):
        PID_X = np.array([self.P_COEFF_FOR[0], self.I_COEFF_FOR[0], self.D_COEFF_FOR[0]])
        PID_Y = np.array([self.P_COEFF_FOR[1], self.I_COEFF_FOR[1], self.D_COEFF_FOR[1]])
        PID_Z = np.array([self.P_COEFF_FOR[2], self.I_COEFF_FOR[2], self.D_COEFF_FOR[2]])
        PID_ROLL = np.array([self.P_COEFF_TOR[0], self.I_COEFF_TOR[0], self.D_COEFF_TOR[0]])
        PID_PITCH = np.array([self.P_COEFF_TOR[1], self.I_COEFF_TOR[1], self.D_COEFF_TOR[1]])
        PID_YAW = np.array([self.P_COEFF_TOR[2], self.I_COEFF_TOR[2], self.D_COEFF_TOR[2]])
        return [PID_X, PID_Y, PID_Z,
                PID_ROLL, PID_PITCH, PID_YAW]
    ################################################################################

    def computeControl(self,
                       vel_ctrl,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Abstract method to compute the control action for a single drone.

        It must be implemented by each subclass of `BaseControl`.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        """
        raise NotImplementedError
