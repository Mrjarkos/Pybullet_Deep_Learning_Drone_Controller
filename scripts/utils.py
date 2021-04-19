import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.fftpack import fft, fftfreq

def euler_to_quaternion(roll, pitch, yaw):
    return (Rotation.from_euler('XYZ',(roll, pitch, yaw), degrees=False)).as_quat()

def plot_fourier(y, fs):
    dt = 1/fs
    n = len(y)
    t = np.arange(0, n*dt, dt)
    Y = fft(y) / n # Transformada normalizada
    frq = fftfreq(n, dt)
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(t, y)
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('$y(t)$')
    ax1.set_title('Señal en el tiempo ref')
    ax2 = fig.add_subplot(212)
    ax2.set_title('Señal en frecuencia ref')
    ax2.vlines(frq[0:int(n/2)], 0, abs(Y[0:int(n/2)]))
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Abs($Y$)')
    return fig
