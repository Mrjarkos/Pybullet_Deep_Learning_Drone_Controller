import os
import time
import math
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock
from DashPlotPSO import DashPlotPSO
from numpy.core.fromnumeric import argmin

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s]  | %(module)s | (%(threadName)-10s) | {%(asctime)s} | %(message)s',
                    #format='[%(levelname)s] |\t| {%(asctime)s} | %(message)s',
                    )
logging.getLogger('matplotlib.font_manager').disabled = True

X_AXIS_NAMES = ['X', 'Y', 'Z', 'W']

class Particle(object):
    def __init__(self, x, v, problem='min'):
        self.position=x
        self.speed=v
        self.bestPosition=x
        self.problem = problem
        if problem == 'min':
            self.bestValue = np.inf
            self.value = np.inf
        else:
            self.bestValue = -np.inf
            self.value = -np.inf
    
    def evaluate(self, costFunction):
        value=costFunction(self.position)
        
        #Update Personal Best Value
        if self.problem == 'min':
            if value < self.bestValue:
                self.bestValue = value
                self.bestPosition = self.position
        else:
            if value > self.bestValue:
                self.bestValue = value
                self.bestPosition = self.position
        self.value=value
        
    def updateSpeed(self, bestSwarmPos, C, exploration=False, epsilon=0):
        rdn = np.random.rand(2)
        #Update Speed
        if exploration:
            if np.random.rand()<epsilon:
            # Exploration vs Explotation
                self.speed = C[0]*self.speed*4*(np.random.rand(len(self.speed))-0.5)
            else:
                self.speed = C[0]*self.speed+\
                             C[1]*rdn[0]*(self.bestPosition-self.position)+\
                             C[2]*rdn[1]*(bestSwarmPos-self.position)
        else: 
            self.speed = C[0]*self.speed+\
                         C[1]*rdn[0]*(self.bestPosition-self.position)+\
                         C[2]*rdn[1]*(bestSwarmPos-self.position)+\
                         epsilon*self.speed*np.random.rand(len(self.speed))
        
    def updateSpeedRepulsion(self, k, d, l, closestParPos):
        dist = np.array(closestParPos)-np.array(self.position)
        euclideanDist = np.linalg.norm(dist)      # Euclidean Distance
        if euclideanDist<=l:
            self.speed = self.speed -k*abs(euclideanDist-d)*(dist)

    def trimSpeedLimits(self, vel_min, vel_max):
        #Trim speed between limits
        self.speed = np.maximum(np.minimum(self.speed,vel_max), vel_min)

    def updatePos(self, dt):
        #Update Position
        self.position=self.position + np.array(self.speed)*dt

    def mirrorSpeed(self, x_min, x_max):
        #Velocity Mirror Effect whe outside limits
        for i, x in enumerate(self.position):
            if x<x_min[i] or x>x_max[i]:
                self.speed[i]*=-1

    def trimPositionLimits(self, x_min, x_max):
        #Trim position between limits
        self.position = np.maximum(np.minimum(self.position, x_max), x_min)

    def getPosition(self):
        return self.position

    def getValue(self):
        return self.value

class PSO(object):
    def __init__(self, x_lim, costFunction, 
                x_name = [], N=50, dt=0.1, 
                epochs=100, delta=1e-3, patience=5,
                early_stopping = False,
                w_Damping=[],
                C=[1, 0.4, 0.6],
                exploration = False,
                epsilon_Damping=[],
                repulsion_params = [],
                problem='min', verbose=1,
                workers=1, checkpoints=False,
                check_step=1,
                path="./logs/PSOCheckpoints", 
                init_type='uniform'):

        ## Checkpoints
        self.checkpoints = checkpoints                  # Save partial models (checkpoints)
        self.check_step = check_step                    # Each epochs save partial model
        self.path = path                                # Path where model is saved
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ## Problem Definition
        self.costFunction = costFunction                # Cost Function
        self.problem = problem                          # Problem Optimization max or min
        self.dt = dt                                    # Delta Time
        self.init_type = init_type                      # Kind of particle optimization

        # Variable Limits
        self.x_max = np.array(x_lim[0])                 # Upper Bound of Variables
        self.x_min = np.array(x_lim[1])                 # Lower Bound of Variables
        self.v_max = self.dt*(self.x_max-self.x_min)/2  # Upper Bound of Velocity
        self.v_min = -self.v_max                        # Lower Bound of Velocity

        self.n_var = len(self.x_max)                    # Number of Decision Variables
        self.x_name = []                                # Name of Decision Variables
        if len(x_name)==0 and self.n_var<=len(X_AXIS_NAMES):
            for i in range(self.n_var):
                self.x_name.append(X_AXIS_NAMES[i])
        else:
            self.x_name = x_name

        ## PSO Parameters
        self.epochs = epochs                            # Maximum Number of Iterations
        self.N = N                                      # Number of Particles
        self.w_Damping = w_Damping                      # W damping parameters initial, final, ratio
        self.C = C                                      # Inertia, Personal Learning, Global Learning, Random Speed
        self.exploration = exploration                  # Whether to use Exploration factor or just exploitation
        self.epsilon_Damping = epsilon_Damping          # Exploration Factor Damping initial, final, ratio
        if len(epsilon_Damping)>0:
            self.epsilon = epsilon_Damping[0]           # Initial Exploration Factor
        else: 
            self.epsilon = 0
        if len(repulsion_params)>0:
            self.repulsion=True
            self.repulsion_params = repulsion_params
        else:
            self.repulsion=False
            self.repulsion_params = [0]*3
        ## Early Stopping
        self.early_stopping = early_stopping            # Whether to use early_stopping or not
        self.delta = delta                              # Minimum cost variance to stop the algorithm
        self.patience = patience                        # Number of epochs under delta cost variation
        
        ## Aditional Parameters
        self.verbose = verbose                          # Printing and Plotting
        if workers<=N:                                  # Parallel Processing
            self.workers = workers
        else:
            self.workers=N
        self.mutex = Lock()                             # Multiprocessing synchronization

        ## Logging
        if verbose==1 or verbose>2:
            logging.info(f"Swarm created --------------- --------------- ---------------")
            logging.info(f"Max Epochs: {self.epochs}")
            if self.early_stopping:
                logging.info(f"Early Stopping Activated")
                logging.info(f"Delta Cost: {self.delta}")
                logging.info(f"Patience: {self.patience}")        
            logging.info(f"Number of Particles: {self.N}")
            logging.info(f"Coefficients C [Inertia, Personal Learning, Global Learning, Exploration]: {self.C}")
            logging.info(f"Inertia Damping Coefficients W [Initial, Final, Ratio, Exploration factor]: {self.w_Damping}")
            logging.info(f"Number of Decision Variables: {self.n_var}")
            logging.info(f"Names of Decision Variables: {self.x_name}")
            logging.info(f"Limits of Decision Variables: {self.x_max}, {self.x_min}")
            logging.info(f"Limits of Speeds: {self.v_max}, {self.v_min}")
            if self.repulsion:
                logging.info(f"Repulsion Activated")
                logging.info(f"Repulsion factor: {self.repulsion_params[0]}")
                logging.info(f"Comfortable distance: {self.repulsion_params[1]}")
                logging.info(f"Idle distance: {self.repulsion_params[2]}")        

    def initializeParticles(self, type='uniform'):
        logging.info(f"Initializing Particles --------------- --------------- ---------------")
        logging.info(f"Initialization method: {type}")
        self.particles = []                                          # Particle List
        self.HistVal = np.empty([self.epochs])                       # History of Current Value per Particle
        self.HistBestVal = np.empty([self.epochs])                   # History of Best Value Ever
        self.HistBestPos = np.empty([self.epochs, self.n_var])       # History of Best Particle Position Ever
        self.HistParPos = np.empty([self.epochs,self.N,self.n_var])  # History of Current Position per Particle
        self.HistParVal = np.empty([self.epochs,self.N])             # History of Value per Particle
        self.HistW = []                                              # History of Inertia Parameters
        self.HistEpsilon = []                                        # History of Inertia Parameters
        self.deltaCost = np.inf                                      # Initial Delta Cost
        if self.problem == 'min':                                    # Best Value All Iterations (Solution)
            self.bestValEver = np.inf
        else:
            self.bestValEver = -np.inf
        self.bestPosSwarm=[]                                         # Best Swarm Position per iteration
        
        x = []
        n = round(self.N**(1/self.n_var))                            # Number of particles per dimension
        self.workstep = math.floor(len(self.particles)/self.workers)
        step = (self.x_max-self.x_min)/n
        if self.repulsion:
            self.repulsion_params[1] = self.repulsion_params[1]*step
            self.repulsion_params[2] = self.repulsion_params[2]*min(step)
        init_pos_par = []
        for i in range(self.n_var):
            init_pos_par.append(np.arange(self.x_min[i]+step[i]/2, self.x_max[i], step[i]))
        if type=='uniform':                                          # Uniform Initialization
            for combination in itertools.product(*init_pos_par):
                x.append(np.asarray(combination))
        elif type=='random':                                         # Random Initialization
            for _ in range(self.N):
                x.append(np.random.rand(self.n_var)*(self.x_max-self.x_min)+self.x_min)
        else:                                                        # Uniform + Random Initialization
            for combination in itertools.product(*init_pos_par):
                x.append(
                    np.maximum(
                        np.minimum(
                            np.asarray(combination)+\
                            step*(
                                np.random.rand(self.n_var)*(self.x_max-self.x_min)+self.x_min
                                ),
                            self.x_max
                        ),
                        self.x_min
                    )
                )

        if len(x)<self.N:                                           # When the number of particles isn't exact
            for i in range(self.N-len(x)):
                x.append(np.random.rand(self.n_var)*(self.x_max-self.x_min)+self.x_min)
        x=np.array(x)
        
        # Initialize Particles
        for i in range(self.N):
            self.particles.append(  Particle( x = x[i], 
                                              v = np.random.rand(self.n_var)*(self.v_max-self.v_min)+self.v_min, 
                                              problem=self.problem
                                            )
                                    )

    def evalParticle(self, j, N, n):
        # Evaluate Particle
        if j<n:
            ini = j*(self.workstep+1)
            final = ini+(self.workstep+1)
        else:
            ini = j*(self.workstep+1)+(n-j)
            final = ini+self.workstep
        if j==N-1:
            final=len(self.particles)
        for i in range(ini, final):
            self.particles[i].evaluate(self.costFunction)   # Call cost function
            self.mutex.acquire()                            # Multithread Synchronization
            self.values[i] = self.particles[i].getValue()   # Add value to value list
            self.mutex.release()

    def evaluateParticles(self):
        # Evaluate all Particles
        self.values = np.empty([len(self.particles)])
        threads = []
        for i in range(self.workers):
            t = Thread(target=self.evalParticle, name=f'{i}', args=(i,self.workers,self.N%self.workers))
            threads.append(t)
            t.start()        
        for t in threads:
            t.join()

    def findBestPos(self):
        if len(self.bestPosSwarm)==0:
            self.bestPosSwarm = self.particles[0].getPosition()
        # Find Episode Best Swarm Position
        if self.problem == 'min':
            self.bestValSwarm = np.amin(self.values)
            if self.bestValSwarm<self.bestValEver:
                self.bestValEver = self.bestValSwarm
                self.bestPosEver = self.bestPosSwarm
        else:
            self.bestValSwarm = np.amax(self.values)
            if self.bestValSwarm>self.bestValEver:
                self.bestValEver = self.bestValSwarm
                self.bestPosEver = self.bestPosSwarm
        index = np.where(self.values == self.bestValSwarm)
        if len(index) > 0 and len(index[0]) > 0:
            index = index[0][0]
        self.bestPosSwarm = self.particles[index].getPosition()
        
    def getClosestParticlePos(self, particle, positions):
        dist = np.empty([len(positions)])
        for i, p in enumerate(positions):
            dist[i] = np.linalg.norm(np.array(particle)-np.array(p))       # Euclidean Distance
        minim = sorted(dist)[1]
        return positions[ np.where(dist == minim)[0][0]]

    def step(self):
        positions = []
        for p in self.particles:
            positions.append(p.getPosition())
        for p in self.particles:
            p.updateSpeed(self.bestPosSwarm, self.C, self.exploration, self.epsilon)
            best = (p.getValue()==self.bestValSwarm)
            if self.repulsion and not best:
                p.updateSpeedRepulsion(self.repulsion_params[0],
                                       self.repulsion_params[1],
                                       self.repulsion_params[2], 
                                       self.getClosestParticlePos(p.getPosition(), positions))
            p.trimSpeedLimits(self.v_min, self.v_max)
            p.updatePos(self.dt)
            p.mirrorSpeed(self.x_min, self.x_max)
            p.trimPositionLimits(self.x_min, self.x_max)

    def updateInertia(self, i):
        self.C[0] = (self.w_Damping[0]-self.w_Damping[1])*(1-(i/self.epochs)**self.w_Damping[2])+self.w_Damping[1]

    def updateEpsilon(self, i):
        self.epsilon = (self.epsilon_Damping[0]-self.epsilon_Damping[1])*(1-(i/self.epochs)**self.epsilon_Damping[2])+self.epsilon_Damping[1]

    def EarlyStopping(self, i):
        if i>1:
            self.deltaCost = abs(self.HistVal[i]-self.HistVal[i-1])
        if self.deltaCost<self.delta:
            self.d+=1
            logging.debug(f"Delta Cost: {self.deltaCost}, Consecutive:{self.d}")
        if self.d>self.patience:
            logging.info(f"Stopping algorithm by Early Stopping Condition")
            return True
        return False

    def Log(self, i):
        self.HistVal[i] = self.bestValSwarm
        self.HistBestVal[i] = self.bestValEver
        self.HistBestPos[i] = self.bestPosEver
        for n in range(self.N):
            self.HistParPos[i, n] = self.particles[n].getPosition()
            self.HistParVal[i, n] = self.particles[n].getValue()
        self.HistW.append(self.C[0])
        self.HistEpsilon.append(self.epsilon)
        if self.verbose==1 or self.verbose>2:
            logging.info(f"Epoch = {i}, Processing Time={round(time.time() - self.start_time,3)}s, bestValSwarm = {self.bestValSwarm}, bestPosSwarm = {self.bestPosSwarm}")
        
        self.history = { 'i': i,
                   'values': self.HistVal,
                   'bestValue':self.HistBestVal,
                   'bestPosition':self.HistBestPos,
                   'particlePosition': self.HistParPos,
                   'particleValue': self.HistParVal,
                   'inertia':self.HistW,
                   'exploration': self.HistEpsilon,
                   'parameters': {
                       'x_names': self.x_name,
                       'x_min': self.x_min,
                       'x_max': self.x_max,
                       'n_var': self.n_var,
                       'exploration': self.exploration,
                       'N': self.N
                   }
        }
        self.solution = { 'value':    self.bestValEver,
                'solution': self.bestPosEver,
                'history': self.history
        }

    def checkpoint(self, i):
        try:
            if i>0:
               with open(f'{self.path}/checkpoint_{i-1}.npy', 'rb') as f:
                    solution = np.load(f)
                    bestValSwarm = solution['value']
                    bestPosSwarm = solution['solution']
                    if (bestValSwarm<self.bestValEver and self.problem=='min') or \
                    (bestValSwarm>self.bestValEver and self.problem=='max'):
                        return 0
        except:
            pass
        finally:
            with open(f'{self.path}/checkpoint_{i}.npy', 'wb') as f:
                np.save(f, self.solution)

    def optimize(self):
        logging.info(f"Executing PSO Algorithm with {self.workers} workers")
        self.n = 0 
        self.d = 0
        self.initializeParticles(self.init_type)       # Initialize Particles
        self.start_ini_time = time.time()
        j=0
        for i in range(self.epochs):
            self.start_time = time.time()
            self.evaluateParticles()
            self.findBestPos()
            self.step()
            self.Log(i)
            if len(self.w_Damping)>0:
                self.updateInertia(i)
            if len(self.epsilon_Damping)>0:
                self.updateEpsilon(i)
            self.n = i 
            if self.checkpoints and i%self.check_step==0:
                print(self.checkpoints)
                self.checkpoint(j)
                self.j+=1
            if self.early_stopping:
                if self.EarlyStopping(i):
                    break
        self.checkpoint(j)
        if self.verbose>3:
            self.Dashplot(j)
        elif self.verbose==2 or self.verbose==3:
            self.plot()
        logging.info(f'TotalTime: {round(time.time() - self.start_ini_time,3)}s, Answer: bestValSwarm = {self.bestValEver}, bestPosSwarm = {self.bestPosEver}')
        return self.solution

    def plot(self):
        cname = 'summer'
        if self.verbose==2 or self.verbose==3:
            _,ax = plt.subplots()
            lns1 = ax.plot(self.HistVal[:self.n], color="red", label='Cost Value')
            lns2 = ax.plot(self.HistBestVal[:self.n], color="green", label='Best Cost Value')
            ax.grid()
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Cost per Iteration")
            ax1=ax.twinx()
            lns3 = ax1.plot(self.HistW[:self.n], color="blue", label='Inertia Value')
            if self.exploration:
                lns4 = ax1.plot(self.HistEpsilon[:self.n], color="yellow", label='Exploration Value')
                lns3+=lns4
            ax1.set_ylabel("Inertia Value")
            lns = lns1+lns2+lns3
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)
            if self.n_var==2:
                fig=plt.figure(figsize=(10,10))
                ax2=fig.add_subplot(1,1,1)
                for n in range(self.N):
                    ax2.scatter(self.HistParPos[:,n,0],
                                self.HistParPos[:,n,1],
                                c=self.HistParVal[:,n],
                                cmap=plt.get_cmap(cname),
                                alpha = 0.4,
                                s=10,
                                marker='.')
                ax2.scatter(self.HistParPos[0,:,0],
                            self.HistParPos[0,:,1], 
                            c=self.HistParVal[0,:],
                            cmap=plt.get_cmap(cname),
                            s=50,
                            marker='o')
                sctt = ax2.scatter(self.HistParPos[self.n,:,0],
                                   self.HistParPos[self.n,:,1],
                                   c=self.HistParVal[self.n,:],
                                   cmap=plt.get_cmap(cname),
                                   s=50,
                                   marker='X')   
                
                cbar = fig.colorbar(sctt, ax = ax2, shrink = 1, aspect = 10)
                cbar.ax.set_xlabel('Cost Value')
                ax2.set_xlabel(self.x_name[0])
                ax2.set_ylabel(self.x_name[1])
                ax2.set_xlim(self.x_min[0], self.x_max[0])
                ax2.set_ylim(self.x_min[1], self.x_max[1])
                ax2.grid()
                plt.title("PSO Particles Cost")
            elif self.n_var==3:
                fig=plt.figure(figsize=(10,10))
                ax2=plt.axes(projection='3d')
                for n in range(self.N):
                    ax2.scatter3D(self.HistParPos[:,n,0],
                                  self.HistParPos[:,n,1],
                                  self.HistParPos[:,n,2], 
                                  c=self.HistParVal[:,n],
                                  cmap=plt.get_cmap(cname),
                                  alpha = 0.4,
                                  s=10,
                                  marker='.')
                ax2.scatter3D(self.HistParPos[0,:,0],
                              self.HistParPos[0,:,1],
                              self.HistParPos[0,:,2], 
                              c=self.HistParVal[0,:],
                              cmap=plt.get_cmap(cname),
                              s=50,
                              marker='o')
                sctt = ax2.scatter3D(self.HistParPos[self.n,:,0],
                                    self.HistParPos[self.n,:,1],
                                    self.HistParPos[self.n,:,2],
                                    c=self.HistParVal[self.n,:],
                                    cmap=plt.get_cmap(cname),
                                    s=50,
                                    marker='X')   

                cbar = fig.colorbar(sctt, ax = ax2, shrink = 0.8, aspect = 10)
                cbar.ax.set_xlabel('Cost')
                ax2.set_xlabel(self.x_name[0])
                ax2.set_ylabel(self.x_name[1])
                ax2.set_zlabel(self.x_name[2])
                ax2.set_xlim(self.x_min[0], self.x_max[0])
                ax2.set_ylim(self.x_min[1], self.x_max[1])
                ax2.set_zlim(self.x_min[2], self.x_max[2])
                ax2.grid()
                plt.title("PSO Particles Cost")
            plt.show(block=False)

    def Dashplot(self, j):
        DashPlotPSO(f'{self.path}/checkpoint_{j}.npy')
        #plot.plot()