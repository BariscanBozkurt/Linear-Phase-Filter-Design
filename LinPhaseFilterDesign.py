"""
Title: LinPhaseFilterDesign.py

Explanation: Designing an FIR filter with given pass & stop bands with the constraints of the optimization problem.

Code Writer: Barışcan Bozkurt (Koç University - EEE & Mathematics)

Date: 26.06.2020
"""


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class LinPhaseFilterDesign():
    def __init__(self,N=20,phase='linear',norm=1,W=False,penalize=False):
        self.N=N
        self.phase=phase
        self.norm=1
        self.W=W
        self.penalize=False
        
    def DBfreq_response(self,x,fft_point=256):
        return 20*np.log10(np.absolute(np.fft.fft(x,fft_point)))[0:round(fft_point/2)]

    def lin_phase_design(self,w,desired, verbose = True):
        n=self.N+1
        F=np.zeros((desired.shape[0],n))
        for i in range(len(w)):
            for j in range(n):
                if j==0:
                    F[i,j]=1
                else:
                    F[i,j]=2*np.cos(w[i]*j)
        h=cp.Variable(n)

        if self.W==True:
            W=np.diag(1/desired) #Generate a diagonal matrix with the reciprocal of the magnitude of desired frequencies

        if self.W==False and self.penalize==False:
            obj=cp.Minimize(cp.norm(desired-F@h,self.norm)) 
        elif self.W==True and self.penalize==False:
            obj=cp.Minimize(cp.norm(W@(desired-F@h),self.norm))

        prob=cp.Problem(obj)
        prob.solve()

        if verbose:
            print('Problem status: {}'.format(prob.status))
            
        filter_=np.concatenate((np.flip(h.value),h.value[1:]))
        self.filter_=filter_
        self.w=w
        self.fft_points=2*desired.shape[0]
        self.desired=desired
        return filter_

    
    def apply_filter(self, signal, mode = 'same'):
        
        filtered_signal = np.convolve(signal, self.filter_, mode = mode)
        
        return filtered_signal
        
    def plot_mag(self,fft_points=False, figheight = 7, figlength = 7):
        if fft_points == False:
            fft_points=self.fft_points

        H=np.absolute(np.fft.fft(self.filter_,fft_points))
        log_H=DBfreq_response(self.filter_,fft_points)
        log_desired=20*np.log10(self.desired)
        fig, ax= plt.subplots(2,1)
        fig.set_figheight(figheight)
        fig.set_figwidth(figlength)
        fig.suptitle('Magnitude Response of the Designed Filter')
        ax[0].plot(self.w,H[0:self.desired.shape[0]],label="Designed Filter")
        ax[0].plot(self.w,self.desired,label="Desired Filter")
        ax[0].title.set_text('FFT Domain Freq Plot of the Designed Filter')
        ax[1].plot(self.w,log_H,label="Designed Filter")
        ax[1].plot(self.w,log_desired,label="Desired Filter")
        ax[1].title.set_text('FFT Log Domain Freq Plot of the Designed Filter')
        ax[0].legend()
        ax[1].legend()
        plt.show()

def db2mag(x):
    return 10**(x/20)

def DBfreq_response(x,fft_point=256):
    return 20*np.log10(np.absolute(np.fft.fft(x,fft_point)))[0:round(fft_point/2)]
