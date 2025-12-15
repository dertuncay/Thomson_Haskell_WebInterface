import numpy as np
from scipy import signal
import math

# FFT
def calc_fft(data,npts,sr):
	'''
	data: time series signal
	npts: number of points in time series
	sr: sampling rate
	'''
	Fdat = np.fft.fft(data,npts)
	freq = np.fft.fftfreq(npts, d=1./sr)
	return Fdat, freq #2.0/sr * np.abs(Fdat)  np.abs(Fdat) 

#iFFT
def calc_ifft(data):
	'''
	data: FFT of a signal
	'''
	return np.fft.ifft(data) #irfft

# Regularization
def regularization(uref,sp=1):
	'''
	uref: FFT of reference station
	sp: The spectral percentage to define epsilon (the regularisation parameter) defined in %
	'''
	average_spectral_power = np.mean(abs(uref) ** 2)
	return  sp/100 * average_spectral_power

# Deconvolution
def deconvolve(uref,usta,sp=1,time_domain=True):
	'''
	uref: FFT of the reference station
	usta: FFT of the signal that is going to be deconvolved
	time_domain: Default (True) if true, return deconvolution in time domain, for not return in frequency domain.
	'''
	fft_deco = usta * np.conj(uref) / (abs(uref)**2 + regularization(uref,sp=sp))
	#deconvolution
	if time_domain:
		return np.real(calc_ifft(fft_deco))
	else:
		return fft_deco

# Seismic interferometry by deconvolution using Tikhonov regularisation
def interfer_tikhonov(decon,sr,r,dstack,time_domain=True):
	'''
	Inputs = 
	decon: deconvolved data
	sr: sampling rate
	r: Factor of signal resampling for better peak picking (signal * r)
	dstact: Duration of the deconvolved signal to stack (in [sec])
	time_domin: Default (True) if true, decon is in time domain, if not it is in frequency domain.
	Outputs = 
	sig_deco_r: output signal in time domain
	t: time axis information
	'''
	# Convert from Frequency domain to time domain
	if time_domain:
		sig_deco = decon
	else:
		sig_deco = np.real(calc_ifft(decon))
	# deconvolved signal length
	nt = len(sig_deco) 
	Fs_r = sr * r
	# flipping the signal
	sig_deco = np.fft.fftshift(sig_deco)
	# signal resampling -> resample(sig, number of samples) --> sig * p/q
	sig_deco = signal.resample(sig_deco, int(nt*r))
	nt2 = len(sig_deco)   # length of the resampled signal
	# deconvolved and resampled signal - taken just the dstack*2 length
	sig_deco = sig_deco[(math.floor(nt2/2)-int(dstack*Fs_r)):(math.floor(nt2/2)+1+int(dstack*Fs_r))]
	# time axis for the interferogram
	t = np.linspace(-dstack, dstack, len(sig_deco))
	return sig_deco, t