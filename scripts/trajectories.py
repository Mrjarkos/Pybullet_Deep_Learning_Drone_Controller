import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def step_ret0(length, max, min, N_cycles, each):
    '''
    Return a cyclical step trajectory with length points data and return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        max (float) -> Upper limit of the path 
        min (float) -> Lower limit of the path
        N_iter (int) -> Number of cycles in the data, important to calculate step size
        each (int) -> How many numbers does the reference change
    '''
    path = np.zeros(length)
    a = 0
    b = 0
    up = True
    step = (max-min)/((length/(2*each))/N_cycles)*0.9
    for i in range(length):
        if int(i%each)==0:
            if a>max:
                up=False
                step*=1.1
            if a<min:
                up=True

            if up:
                a=a+step
                b = np.random.rand()*step
            else:
                a=a-step
                b = (np.random.rand()-1)*step

        if i%each>each/2:
            path[i] = a+b
        else:
            path[i] = 0
        path[i] = np.maximum(np.minimum(path[i], max), min)
    return path

def random_step(length, K, DC, each):
    '''
    Return a random step trajectory with length points data and return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        K (float) -> Upper-Lower limit of the path (Gain or Amplitude)
        DC (float) -> Lower limit of the path (DC Level)
        each (int) -> How many steps does the reference change
    '''
    path = np.zeros(length)
    for i in range(length):
        if int(i%each)==0:
            a = (K-DC)*np.random.rand()+DC

        if i%each>each/2:
            path[i] = a
        else:
            path[i] = 0
    return path

def ramp_step_notret0(length, max, min, N_cycles, each):
    '''
    Return a cyclical step trajectory with length points data and return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        max (float) -> Upper limit of the path 
        min (float) -> Lower limit of the path
        N_iter (int) -> Number of cycles in the data, important to calculate step size
        each (int) -> How many numbers does the reference change
    '''
    path = np.zeros(length)
    a = 0
    b = 0
    up = True
    step = (max-min)/((length/(2*each))/N_cycles)*0.5
    for i in range(length):
        if int(i%each)==0:
            if a>max:
                up=False
            if a<min:
                up=True

            b = np.random.rand()*step*0.1
            
            if up:
                a=a+step
            else:
                a=a-step

        if i%each>each/2:
            path[i] = a+b
        else:
            path[i] = -(a+b)
        path[i] = np.maximum(np.minimum(path[i], max), min)
    return path

def step_notret0(length, max, min, N_cycles, each):
    '''
    Return a cyclical step trajectory with length points data and not return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        max (float) -> Upper limit of the path 
        min (float) -> Lower limit of the path
        N_iter (int) -> Number of cycles in the data, important to calculate step size
        each (int) -> How many numbers does the reference change
    '''
    path = np.zeros(length)
    a = 0
    b = 0
    up = True
    step = (max-min)/((length/(2*each))/N_cycles)*0.9
    for i in range(length):
        if int(i%each)==0:
            if a>max:
                up=False
                step*=1.09
            if a<min:
                up=True

            if up:
                a=a+step
            else:
                a=a-step

            b = (np.random.rand()-0.5)*step
        path[i] = a+b
        path[i] = np.maximum(np.minimum(path[i], max), min)
    return path

def big_step_ret0(length, max, min, N_cycles, each):
    '''
    Return a cyclical step trajectory with length points data and not return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        max (float) -> Upper limit of the path 
        min (float) -> Lower limit of the path
        N_iter (int) -> Number of cycles in the data, important to calculate step size
        each (int) -> How many numbers does the reference change
    '''
    path = np.zeros(length)
    a = 0
    b = 0
    up = True
    step = (max-min)/((length/(2*each))/N_cycles)*0.9
    for i in range(length):
        if int(i%each)==0:
            if a>max:
                up=False
                step*=1.1
            if a<min:
                up=True

            if up:
                a=a+step
                b = 2*np.random.rand()*step
            else:
                a=a-step
                b = 2*(np.random.rand()-0.5)*step

        if i%each<each/4:
            path[i] = a+b
        elif i%each>each/2 and i%each<3*each/4:
            path[i] = -(a+b)    
        else:
            path[i] = 0
        path[i] = np.maximum(np.minimum(path[i], max), min)
    return path

def big_step_notret0(length, max, min, N_cycles, each):
    '''
    Return a cyclical step trajectory with length points data and not return to 0, N_iter cycles and beetwen max and min values.
        length (int) -> number of points of the trajectory
        max (float) -> Upper limit of the path 
        min (float) -> Lower limit of the path
        N_iter (int) -> Number of cycles in the data, important to calculate step size
        each (int) -> How many numbers does the reference change
    '''
    path = np.zeros(length)
    a = 0
    b = 0
    up = True
    step = (max-min)/((length/(2*each))/N_cycles)*0.9
    for i in range(length):
        if int(i%each)==0:
            if a>max:
                up=False
                step*=1.1
            if a<min:
                up=True

            if up:
                a=a+step
                b = 2*np.random.rand()*step
            else:
                a=a-step
                b = 2*(np.random.rand()-0.5)*step

        if i%each>each/2 and i%each<3*each/4:
            path[i] = -(a+b)    
        else:
            path[i] = path[i] = a+b
        path[i] = np.maximum(np.minimum(path[i], max), min)
    return path

def chirp(duration, fs, k, f0, f1, t1, method='linear'):
    '''
    Generate swept-frequency cosine (chirp) signal length points data N_iter cycles and beetwen max and min values.
        duration (float seconds) -> How long is the simulation? 
        fs (float Hz) -> Sampling Freq
        k (int) -> Gain
        f0 (float) -> Frequency (e.g. Hz) at time t=0.
        t1 (float) -> Time at which f1 is specified.
        f1 (float) -> Frequency (e.g. Hz) of the waveform at time t1.
        method {‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}, optional -> Kind of frequency sweep. If not given, linear is assumed. See Notes below for more details.
    '''
    if f0<0.001 or f1<0.001 or t1<0.001 or fs<0.001:
        raise Exception("Sorry, no numbers below zero")
    
    t = np.linspace(0, duration, duration*fs)
    return k*signal.chirp(t, f0=f0, f1=f1, t1=t1, method=method, phi=-90)
     
def triangular_sweep(duration, fs, k, f0, f1):
    '''
    Generate swept-frequency triangular (chirp) signal length points data with gain K.
        duration (float seconds) -> How long is the simulation? 
        fs (float Hz) -> Sampling Freq
        k (int) -> Gain
        f0 (float) -> Frequency (e.g. Hz) at time t=0.
        f1 (float) -> Frequency (e.g. Hz) of the waveform at time t1=duration.
    '''
    if f0<0.001 or f1<0.001 or fs<0.001:
        raise Exception("Sorry, no numbers below zero")
    samples = duration*fs
    t = np.linspace(0,duration,num=samples)
    f_sweep = np.linspace(f0,f1,num=samples) # Sweep from slow to high frequency
    path = k*signal.sawtooth(np.pi*f_sweep*t, width=0.5)
    return path

def sawtooth_sweep(duration, fs, k, f0, f1):
    '''
    Generate swept-frequency triangular (chirp) signal length points data with gain K.
        duration (float seconds) -> How long is the simulation? 
        fs (float Hz) -> Sampling Freq
        k (int) -> Gain
        f0 (float) -> Frequency (e.g. Hz) at time t=0.
        f1 (float) -> Frequency (e.g. Hz) of the waveform at time t1=duration.
    '''
    if f0<0.001 or f1<0.001 or fs<0.001:
        raise Exception("Sorry, no numbers below zero")
    samples = duration*fs
    t = np.linspace(0,duration,num=samples)
    f_sweep = np.linspace(f0,f1,num=samples) # Sweep from slow to high frequency
    path = k*signal.sawtooth(np.pi*f_sweep*t, width=1)
    return path

def stopped(length, DC): 
    return np.zeros(length)+DC

def noise(duration, fs, k, f0, Order=10):
    '''
    White noise with k gain and f0 cut frequency
        duration (float seconds) -> How long is the simulation? 
        fs (float Hz) -> Sampling Freq
        k (int) -> Gain
        f0 (float) -> Cut Frequency (Hz)
        Order (int) -> Filter order
    '''
    if f0<0.001 or fs<0.001 or Order<1:
        raise Exception("Sorry, no numbers below zero")
    samples = duration*fs
    sig = 2*k*(np.random.random(size=samples)-0.5)
    sos = signal.butter(int(Order), 2*np.pi*f0, 'low', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered
    
def step(duration, val):
    path = np.zeros(duration)
    for i in range(duration):
        if i>0.2*duration and i<0.7*duration:
            path[i] = val
    return path

def pulse(duration, val, finish):
    path = np.zeros(duration)
    for i in range(duration):
        if i>0.2*duration and i<finish*duration:
            path[i] = val
    return path

def ramp(duration, m, fs):
    path = np.zeros(duration)
    for i in range(duration):
        if i>0.2*duration and i<0.7*duration:
            path[i] = m*(i/fs-0.2*duration/fs)
    return path

def square(duration, m, fs):
    path = np.zeros(duration)
    for i in range(duration):
        if i>0.2*duration and i<0.7*duration:
            path[i] = m*(i/fs-0.2*duration/fs)**2
    return path

def sin(duration, a, f, fs):
    t = np.linspace(0, duration/fs, duration)
    return a*np.sin(2*np.pi*f*t)