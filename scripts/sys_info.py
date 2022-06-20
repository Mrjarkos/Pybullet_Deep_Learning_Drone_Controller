import numpy as np
import matplotlib.pyplot as plt

N = 5
def step_info(t, yout):
    print("OS: %f%s"%((yout.max()/yout[-1]-1)*100,'%'))
    print("Tr: %fs"%(t[next(i for i in range(0,len(yout)-1) if yout[i]>yout[-1]*.90)]-t[0]))
    print("Ts: %fs"%(t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1])>1.02)]-t[0]))

def overshoot(y, t, y_ts=0):
    flag = False
    if y_ts ==0:
        y_ts = y[-N]
        if abs(y[-N]) <=0.001:
            flag=True
            y_ts = 1
            y = y +1
    i = np.argmax(y)
    os = abs((y[i]-y_ts)/y_ts)
    tp = t[i]
    y_s = y[i]
    if flag:
        y_s = y_s-1
    return os*100, tp, y_s

def settling_time(y, t):    
    try:
        i = next(len(y)-i for i in range(2,len(y)-1) if (abs(y[-i]/y[-N])>1.01 or abs(y[-i]/y[-N])<0.99))
    except:
        i=-N
    ts = t[i]-t[0]
    y_ts = y[i]
    return y_ts, ts

def rise_time(y, t, y_ts=0):
    if y_ts ==0:
        y_ts = y[-N]
        if abs(y[-N]) <=0.001:
            y_ts = 1
            y = y +1
    try:
        i = next(i for i in range(0,len(y)-1) if abs(y[i])>abs(y_ts)*.90)
    except:
        i=0
    tr_f = t[i]
    yr_f = y[i]
    try:
        i = next(i for i in range(0,len(y)-1) if abs(y[i])>abs(y_ts)*.10)
    except:
        i=-1
    tr_i = t[i]
    yr_i = y[i]
    tr = tr_f-tr_i
    return tr, tr_i, yr_i, tr_f, yr_f

def ISTE(t, y, uy):
    e = (y-uy)/(max(uy)) if max(abs(uy))!=0 else (y-uy)
    return sum(t*abs(e))

def MSE(y, uy):
    return 1/len(y)*sum((y-uy)**2)
    
def ess(y, uy):
    return np.abs(y-uy)/np.abs(y)*100, np.abs(y-uy)

def proj_point(x,y, text =''):
    arrowprops={'arrowstyle': '-', 'ls':'-.', 'alpha':0.5}
    plt.annotate(text, xy=(x,y), xytext=(x, 0))
    plt.annotate('', xy=(x,y), xytext=(x, 0), 
                textcoords=plt.gca().get_xaxis_transform(),
                arrowprops=arrowprops,
                va='top', ha='center')
    plt.annotate('', xy=(x,y), xytext=(0, y), 
                textcoords=plt.gca().get_yaxis_transform(),
                arrowprops=arrowprops,
                va='center', ha='right')
    

if __name__ == '__main__':
    from scipy import signal
    import matplotlib.pyplot as plt
    lti = signal.lti([1.0], [1.0, -1.0, -1.0])
    t, y = signal.step(lti)
    y = y - 1
    y_ts, ts = settling_time(y, t)
    y_ts += 1
    tr, tr_i, yr_i, tr_f, yr_f = rise_time(y, t)
    os, tp, y_max = overshoot(y, t)
    print(f'ts = {ts}, ys = {y_ts}')
    print(f'Overshoot = {os*100}%, Peak Time = {tp}')
    print(f'Rise Time = {tr}')

    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Step response for 1. Order Lowpass')
    plt.grid()
    plt.scatter([ts, tp, tr_i, tr_f], [y_ts, y_max, yr_i, yr_f])
    plt.show()