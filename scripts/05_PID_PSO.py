import utils
import Losses
import numpy as np
from os import stat
from PSO import PSO
import trajectories
import multiprocessing
from DronePySim import Drone, PybulletSimDrone 

def pySim(x, trajectories=['step', ], axis=['z'], params=[{'val': 1}], console_out=False, plot=False, duration_sec=10):
    pySim = PybulletSimDrone(console_out=console_out, plot=plot, duration_sec=duration_sec)
    drone = Drone(INIT_XYZS=[0,0,5],
                    INIT_RPYS=[0,0,0],
                    control_timestep=pySim.env.TIMESTEP
                    )
    pySim.setdrones([drone])
    pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
    drone.setPIDParams(PIDparams=[x], PIDaxis=['z'])
    return pySim.runSim()

def pySim_process(x, conn, states=['z', 'uz']):
    logger = pySim(x)
    states = logger.getStates(drone=0, states=states)
    conn.send(states)
    conn.close()
    
def eval(x):
    # parent_conn, child_conn = multiprocessing.Pipe()
    # p = multiprocessing.Process(target=pySim_process, args=(x,child_conn))
    # p.start()
    # states = parent_conn.recv()
    # p.join()
    # return Losses.Huber(states['uz']-states['z'], 500)
    return Losses.Huber(x-0.8, 500)

if __name__ == "__main__":
    PID_base = np.array([1.25, 0.05, 0.5])
    K=np.array([5, 25, 10])
    PDI_sup = PID_base*K
    PDI_inf = PID_base/K
    N = 50
    dt = 0.1
    delta = 1e-5
    epochs = 100
    patience = 10
    w_Damping = [1, 0.4, 0.5]
    C         = [1, 0.4, 0.6]
    epsilon_Damping = [1, 0.3, 0.3]
    repulsion_params = [1, min(PID_base), 1]
    x_lim     = [PDI_sup, PDI_inf] 
    N_Checkpoints = 5
    x_name = ['P','I', 'D']
    pso = PSO(
            N=N,
            dt=dt, 
            epochs=epochs, 
            early_stopping = False,
            delta=delta, 
            patience=patience,
            w_Damping = w_Damping,
            C         = C,
            exploration = True,
            epsilon_Damping = epsilon_Damping,
            x_lim     = x_lim, 
            x_name = x_name,
            repulsion_params = repulsion_params,
            costFunction=eval, verbose=4, problem='min',
            workers = multiprocessing.cpu_count(),
            checkpoints=False, 
            check_step=epochs/N_Checkpoints,
            init_type='uniform'
            )        
    log = pso.optimize()
    error = log['value']
    PID =  log['solution']
    #PID = np.array([10.77418758, 0.18481169, 2.7454838]) ## MSE
    #PID = np.array([3.98098357, 0.09242582, 0.88262833])  ## MSE limitado
    #PID = np.array([3.39282561, 0.076, 0.75747512])      ## Huber MSE
    #PID = np.array([4.46426661, 0.05255871, 0.73536498])   ## Huber MAE
    init_error = pySim(PID_base, plot=True, trajectories=['step'], axis=['z'], params=[{'val': 0.1}])
    pySim(PID, plot=True, trajectories=['step'], axis=['z'], params=[{'val': 0.1}])
    print(F'initial error: {init_error}, PID base: {PID_base}')
    print(F'PSO error: {error}, PID base: {PID}')
    input("Press Enter to exit...") 