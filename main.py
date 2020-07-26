import sys
import os
sys.path.append('/usr/local/lib/python3.7/site-packages')
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def normalize(data):
	return data/np.max(abs(data))

def blackman(x):
	# Blackman window
	a0 = 0.42
	a1 = 0.5
	a2 = 0.08
	N = x.shape[-1]
	n = np.arange(N)
	w = a0 - a1*np.cos(2*np.pi*n/N) - a2*np.cos(4*np.pi*n/N)
	return w*x

def hamming(x):
	# Hamming window
	a0 = 0.53836
	a1 = 0.46164
	N = x.shape[-1]
	n = np.arange(N)
	w = a0 - a1*np.cos(2*np.pi*n/N)
	return w*x


try:
	filename = sys.argv[1]
except:
	print('Error!')
	exit()

if '.wav' not in filename:
	print('Error!')
	exit()



data, fs = sf.read(filename)
num_samples = data.shape[-1]
n = 1024
m = 256
s = 32

zero_thresh = 0.1
min_sparse = 10

snr_db = -100
eps = 1e-8
alpha = 0.1

beta = 0.5

inc = n//2

snr = 10**(snr_db/10)


# making length of data a multiple of n
rem = len(data)%n
data = data[0:len(data)-rem]
data_clean = np.zeros(len(data))


# F is full Fourier matrix
# M is a random sampling matrix
# A is random sensing matrix
rng = np.random.default_rng()
ii = sorted(rng.choice(n, size=m, replace=False))
M = np.zeros((m,n))
j = 0
for i in ii:
	M[j,i] = 1
	j += 1

F = np.array([[ np.exp(2j*np.pi*i*j/n)/n for i in range(n)] for j in range(n)])
A = M.dot(F)

def get_sparse(x):
	# print(len(np.where(np.abs(x) > zero_thresh)[0]))
	x[np.where(np.abs(x) <= zero_thresh)] = 0

	if len(np.where(np.abs(x) > zero_thresh)[0]) < min_sparse:
		return 0

	return np.fft.ifft(x).real
	# return F.dot(x).real


def f(x, y, ii, lm):
	''' Objective function '''
	xt = np.fft.ifft(x)
	vec = xt[ii]
	T1 = np.sum((np.abs(vec - y))**2)
	T2 = np.sum(np.abs(x)**2)/n/snr
	T3 = np.sum(np.abs(x))
	return lm*(T1 - T2) + T3

def shrink(x):
	val = np.zeros(x.shape, dtype=np.complex128)
	for i in range(x.shape[-1]):
		if np.abs(x[i]) > 1e-9:
			val[i] = x[i]/abs(x[i])
	return val


def grad_f_x(x, y, ii, lm):
	''' Gradient of objective, lm ==> lagrange multiplier '''
	xt = np.fft.ifft(x)
	vec = xt[ii]
	res = vec - y
	vec = np.zeros(x.shape, dtype=np.complex128)
	vec[ii] = res
	vec = np.fft.fft(vec)/n

	return lm*(2*vec - (2/n/snr)*x) + shrink(x)

def grad_f_lm(x, y, ii):
	''' Gradient of objective, lm ==> lagrange multiplier '''
	xt = np.fft.ifft(x)
	vec = xt[ii]

	T1 = np.sum((np.abs(vec- y))**2)
	T2 = np.sum(np.abs(x)**2)/n/snr
	return (T1 - T2)


def bls_x(x, y, ii, lm, del_x, alpha, beta):
	''' Backtracking line search '''
	t = 1;
	ff = f(x, y, ii, lm)
	xx = np.sum(np.abs(del_x)**2)

	while f(x + t*del_x, y, ii, lm) > ff-alpha*t*xx:
		t = beta * t
	return t

def bls_lm(x, y, ii, lm, del_lm, alpha, beta):
	''' Backtracking line search '''
	t = 1;
	ff = f(x, y, ii, lm)
	ll = del_lm**2

	while f(x, y, ii, lm + t*del_lm) < ff + alpha*t*ll:
		t = beta * t
	return t

# Looking at each frame
i=0
while i < len(data)-n:
	y = M.dot(hamming(data[int(i):int(i+n)]))
	x = np.fft.fft(hamming(data[int(i):int(i+n)]))
	lm = 0
	# Primal-Dual Gradient Descent with Line Search for each frame
	step = 0
	t1 = t2 = 1000

	while step < 1000: #and t1 > 1e-9:
		step += 1
		print('Frame: {} Step: {}'.format(int(i*(n/inc)/n), step), end='\r')

		del_x = -grad_f_x(x, y, ii, lm)
		# print('Step: {}, frac change: {}, Thresh: {}, t: {}, lm: {}'.format(step, np.max(np.abs(t1*del_x)), eps, t1, lm), end='\n')		
		# print(exit_condition)

		t1 = bls_x(x, y, ii, lm, del_x, alpha, beta)

		if np.max(np.abs((t1*del_x))) < eps:
			break
		x = x + t1*del_x

		del_lm = grad_f_lm(x, y, ii)
		t2 = bls_lm(x, y, ii, lm, del_lm, alpha, beta)

		if lm + t2*del_lm >= 0:
			lm = lm + t2*del_lm

	print('Frame: {} Steps: {}'.format(int(i*(n/inc)/n), step), end='\n')
	# print('LM: {}'.format(lm))
	data_clean[int(i):int(i+n)] += get_sparse(x)
	i += inc


write('out_{}.wav'.format(filename.split('.wav')[0]), fs, 0.7*normalize(data_clean))
name = filename.split('.wav')[0]
plt.figure(name)
tt = np.arange(data_clean.size)/fs
plt.title('Input signal SNR: {}dB'.format(name.split('_')[1]))
plt.plot(tt, normalize(data), label='input')
plt.plot(tt, normalize(data_clean), alpha=0.7, label='output')
plt.ylabel('Normalised Signal value')
plt.legend(loc='upper right')
plt.xlabel('time')
plt.grid(True)
plt.show()









