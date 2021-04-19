import utils
import Losses
import numpy as np
from os import stat
from PSO import PSO
import trajectories
import multiprocessing
from DronePySim import Drone, PybulletSimDrone 

def pySim(x, trajectories=['step'], axis=['z'], params=[{'val': 1}], console_out=False, plot=False, duration_sec=10):
    pySim = PybulletSimDrone(console_out=console_out, plot=plot, duration_sec=duration_sec)
    drone = Drone(INIT_XYZS=[0,0,5],
                    INIT_RPYS=[0,0,0],
                    control_timestep=pySim.env.TIMESTEP
                    )
    pySim.setdrones([drone])
    pySim.initTrajectory(trajectories=trajectories, axis=axis, params=params)
    drone.setPIDParams(PIDparams=[x], PIDaxis=['z'])
    return pySim.runSim()

def pySim_process(x, conn):
    logger = pySim(x)
    states = logger.getStates(drone=0, states=['z', 'uz'])
    conn.send(states)
    conn.close()
    
def eval(x):
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=pySim_process, args=(x,child_conn))
    p.start()
    states = parent_conn.recv()
    p.join()
    return Losses.Huber(states['uz']-states['z'], 500)

if __name__ == "__main__":
    PID_base = np.array([1.25, 0.05, 0.5])
    K=5
    PDI_sup = PID_base*K
    PDI_inf = PID_base/K
    init_error = eval(PID_base)
    # N = 80
    # dt = 0.1
    # delta = 1e-5
    # epochs = 150
    # patience = 10
    # w_Damping = [1, 0.4, 0.5, 0.3]
    # C         = [1, 0.4, 0.6, 0.3]
    # x_lim     = [PDI_sup, PDI_inf] 
    # N_Checkpoints = 5
    # pso = PSO(
    #         N=N,
    #         dt=dt, 
    #         epochs=epochs, 
    #         delta=delta, 
    #         patience=patience,
    #         w_Damping = w_Damping,
    #         C         = C,
    #         x_lim     = x_lim, 
    #         costFunction=eval, verbose=3, problem='min',
    #         workers = multiprocessing.cpu_count(),
    #         checkpoints=True, 
    #         check_step=epochs/N_Checkpoints,
    #         init_type='other'
    #         )        
    # error, PID = pso.optimize()
    #PID = np.array([10.77418758, 0.18481169, 2.7454838])
    #PID = np.array([3.98098357, 0.09242582, 0.88262833])
    #PID = np.array([3.39282561, 0.076, 0.75747512])
    #PID = np.array([4.46426661 0.05255871 0.73536498])
    pySim(PID_base, plot=True, trajectories=['step'], axis=['z'], params=[{'val': 0.1}, {'val': 0.2, 'finish':0.2}])
    #pySimTest(PID)
    print(F'initial error: {init_error}, PID base: {PID_base}')
    #print(F'PSO error: {error}, PID base: {PID}')
    input("Press Enter to exit...")
