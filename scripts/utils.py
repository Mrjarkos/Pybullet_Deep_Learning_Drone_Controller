import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.fftpack import fft, fftfreq

def euler_to_quaternion(roll, pitch, yaw):
    return (Rotation.from_euler('XYZ',(roll, pitch, yaw), degrees=False)).as_quat()

def plot_fourier(y, fs, title=''):
    dt = 1/fs
    n = len(y)
    t = np.arange(0, n*dt, dt)
    Y = fft(y) / n # Transformada normalizada
    frq = fftfreq(n, dt)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    
    ax1.plot(t, y)
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('$y(t)$')
    ax1.set_title('Señal en el tiempo')
    ax1.grid()
    plt.subplots_adjust(hspace = 0.75)
    ax2.set_title('Señal en frecuencia')
    ax2.vlines(frq[0:int(n/50)], 0, abs(Y[0:int(n/50)]))
    ax2.set_xlabel('Frecuencia (Hz)')
    ax2.set_ylabel('Abs($Y$)')
    ax2.grid()
    #fig.suptitle(title, fontsize=16)
    return fig

def limit_rad_trajectory(trajectory):
    for i in range(len(trajectory)):
        while trajectory[i]>math.pi or trajectory[i]<-math.pi:
            if trajectory[i]>math.pi:
                trajectory[i]-=2*math.pi
            elif trajectory[i]<-math.pi:
                trajectory[i]+=2*math.pi
    return trajectory

class Counter:
    def __init__(self):
        self.dict = {}
        
    def add(self, item):
        count = self.dict.get(item, 0)
        self.dict[item] = count + 1
        
    def counts(self, desc=None):
        result = [(val, key) for key, val in self.dict.items()]
        result.sort()
        if desc: result.reverse()
        return result
    
    def total(self):
        result = [val*key for key, val in self.dict.items()]
        return sum(result)

    def mean(self):
        result = [val*key for key, val in self.dict.items()]
        vals = [val for key, val in self.dict.items()]
        return sum(result)/sum(vals)
    
