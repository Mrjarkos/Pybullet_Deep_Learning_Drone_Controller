import time
import math
import logging
import itertools
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Lock

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )
logging.getLogger('matplotlib.font_manager').disabled = True

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
        
    def updateSpeed(self, bestSwarmPos,C):
        rdn = np.random.rand(2)

        #Update Speed
        self.speed = C[0]*self.speed+\
                     C[1]*rdn[0]*(self.bestPosition-self.position)+\
                     C[2]*rdn[1]*(bestSwarmPos-self.position)+\
                     C[3]*np.random.rand(len(self.speed))

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
                N=50, dt=0.1, 
                epochs=100, delta=1e-3, patience=5,
                w_Damping=[1,0.4,0.5, 0.01],
                C=[1, 0.4, 0.6, 0.2],
                problem='min', verbose=1,
                workers=1, checkpoints=False,
                check_step=1,
                path="./logs/PSOCheckpoints", 
                init_type='uniform'):

        # Error control
        if costFunction==None:
            raise TypeError("Cost Function is not set")
        if x_lim==None:
            raise TypeError("Decision variables limits are not setted")
            
        ## Aditional Parameters
        self.verbose = verbose           # Printing and Plotting
        self.workers = workers           # Parallel Processing
        self.mutex = Lock()              # Multiprocessing synchronization

        ## Checkpoints
        self.checkpoints = checkpoints   # Save partial models (checkpoints)
        self.check_step = check_step     # Each epochs save partial model
        self.path = path                 # Path where model is saved

        ## Problem Definition
        self.costFunction = costFunction # Cost Function
        self.problem = problem           # Problem Optimization max or min
        self.dt = dt                     # Delta Time
        self.init_type = init_type       # Kind of particle optimization

        # Variable Limits
        self.x_max = np.array(x_lim[0])                 # Upper Bound of Variables
        self.x_min = np.array(x_lim[1])                 # Lower Bound of Variables
        self.v_max = self.dt*(self.x_max-self.x_min)/2  # Upper Bound of Velocity
        self.v_min = -self.v_max                        # Lower Bound of Velocity

        self.n_var = len(self.x_max)     # Number of Decision Variables

        ## PSO Parameters
        self.epochs = epochs             # Maximum Number of Iterations
        self.N = N                       # Number of Particles
        self.delta = delta               # Minimum cost variance to stop the algorithm
        self.patience = patience         # Number of epochs under delta cost variation
        self.w_Damping = w_Damping       # W damping parameters initial, final, ratio
        self.C = C                       # Inertia, Personal Learning, Global Learning, Random Speed

    def initializeParticles(self, type='uniform'):
        self.particles = []                                          # Particle List
        self.HistVal = np.empty([self.epochs])                       # History of Current Value per Particle
        self.HistBestVal = np.empty([self.epochs])                   # History of Best Value Ever
        self.HistParPos = np.empty([self.epochs,self.N,self.n_var])  # History of Current Position per Particle
        self.HistParVal = np.empty([self.epochs,self.N])             # History of Value per Particle
        self.HistW = []                                              # History of Inertia (and Random) Parameters
        self.deltaCost = np.inf                                      # Initial Delta Cost
        if self.problem == 'min':                                    # Best Value All Iterations (Solution)
            self.bestValEver = np.inf
        else:
            self.bestValEver = -np.inf
        self.bestPosSwarm=[]                                         # Best Swarm Position per iteration
        
        x = []
        n = round(self.N**(1/self.n_var))# Number of particles per dimension
        step = (self.x_max-self.x_min)/n
        init_pos_par = []
        for i in range(self.n_var):
            init_pos_par.append(np.arange(self.x_min[i]+step[i]/2, self.x_max[i], step[i]))
        if type=='uniform':             # Uniform Initialization
            for combination in itertools.product(*init_pos_par):
                x.append(np.asarray(combination))
        elif type=='random':            # Random Initialization
            for _ in range(self.N):
                x.append(np.random.rand(self.n_var)*(self.x_max-self.x_min)+self.x_min)
        else:                           # Uniform + Random Initialization
            for combination in itertools.product(*init_pos_par):
                x.append(np.asarray(combination)+step*(np.random.rand(self.n_var)*(self.x_max-self.x_min)+self.x_min))

        if len(x)<self.N:               # When the number of particles isn't exact
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

    def evalParticle(self, j, N):
        # Evaluate Particle
        step = len(self.particles)//N
        for i in range(step*j, step*(j+1)):
            self.particles[i].evaluate(self.costFunction)   # Call cost function
            self.mutex.acquire()                            # Multithread Synchronization
            self.values[i] = self.particles[i].getValue()   # Add value to value list
            self.mutex.release()

    def evaluateParticles(self):
        # Evaluate all Particles
        self.values = np.empty([len(self.particles)])
        threads = []
        for i in range(self.workers):
            t = Thread(target=self.evalParticle, name=f'{i}', args=(i,self.workers,))
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
        
    def step(self):
        for i, p in enumerate(self.particles):
            self.particles[i].updateSpeed(self.bestPosSwarm, self.C)
            self.particles[i].trimSpeedLimits(self.v_min, self.v_max)
            self.particles[i].updatePos(self.dt)
            self.particles[i].mirrorSpeed(self.x_min, self.x_max)
            self.particles[i].trimPositionLimits(self.x_min, self.x_max)

    def updateInertia(self, i):
        self.C[0] = (self.w_Damping[0]-self.w_Damping[1])*(1-(i/self.epochs)**self.w_Damping[2])+self.w_Damping[1]
        self.C[3] = self.C[0]*self.w_Damping[3]

    def EarlyStopping(self, i):
        if i>1:
            self.deltaCost = abs(self.HistVal[i]-self.HistVal[i-1])
        if self.deltaCost<self.delta:
            self.d+=1
        if self.d>self.patience:
            return True
        return False

    def Log(self, i):
        self.HistVal[i] = self.bestValSwarm
        self.HistBestVal[i] = self.bestValEver
        for n in range(self.N):
            self.HistParPos[i, n] = self.particles[n].getPosition()
            self.HistParVal[i, n] = self.particles[n].getValue()
        self.HistW.append(self.C[0])
        if self.verbose==1 or self.verbose==3:
            print(f'i = {i}, time={round(time.time() - self.start_time,3)}s, bestValSwarm = {self.bestValSwarm}, bestPosSwarm = {self.bestPosSwarm}')

    def checkpoint(self, i):
        try:
            if i>0:
                with open(f'{self.path}/checkpoint_{i-1}.npy', 'rb') as f:
                    bestValSwarm = np.load(f)
                    bestPosSwarm = np.load(f)
                    if (bestValSwarm<self.bestValEver and self.problem=='min') or \
                    (bestValSwarm>self.bestValEver and self.problem=='max'):
                        return 0
        except:
            pass
        finally:
            with open(f'{self.path}/checkpoint_{i}.npy', 'wb') as f:
                np.save(f, self.bestValEver)
                np.save(f, self.bestPosEver)

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
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
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
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('Z')
                ax2.set_xlim(self.x_min[0], self.x_max[0])
                ax2.set_ylim(self.x_min[1], self.x_max[1])
                ax2.set_zlim(self.x_min[2], self.x_max[2])
                ax2.grid()
                plt.title("PSO Particles Cost")
            plt.show(block=False)

    def optimize(self):
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
            self.updateInertia(i)
            self.n = i 
            if self.checkpoints and i%self.check_step==0:
                self.checkpoint(j)
                j+=1
            if self.EarlyStopping(i):
                break
        self.checkpoint(j)
        self.plot()
        print(f'TotalTime: {round(time.time() - self.start_ini_time,3)}s, Answer: bestValSwarm = {self.bestValEver}, bestPosSwarm = {self.bestPosEver}')
        return self.bestValEver, self.bestPosEver