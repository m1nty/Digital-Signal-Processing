import matplotlib.pyplot as plt
import numpy as np
import cmath, math

def plotSpectrum(x, Fs, type = 'FFT'):

	n = len(x)             # length of the signal
	df = Fs/n              # frequency increment (width of freq bin)

	# compute Fourier transform, its magnitude and normalize it before plotting
	if type == 'FFT':
		Xfreq = np.fft.fft(x)
	XMag = abs(Xfreq)/n

	# Note: because x is real, we keep only the positive half of the spectrum
	# Note also: half of the energy is in the negative half (not plotted)
	XMag = XMag[0:int(n/2)]

	# freq vector up to Nyquist freq (half of the sample rate)
	freq = np.arange(0, Fs/2, df)

	fig, ax = plt.subplots()
	ax.plot(freq, XMag)
	ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude',
		title='Frequency domain plot')
	# fig.savefig("freq.png")
	plt.show()

def plotTime(x, time):

    fig, ax = plt.subplots()
    ax.plot(time, x)
    ax.set(xlabel='Time (sec)', ylabel='Amplitude',
            title='Time domain plot')
    # fig.savefig("time.png")
    plt.show()

def generateSin(Fs, interval, frequency = 7.0, amplitude = 5.0, phase = 0.0):

    dt = 1.0/Fs                          # sampling period (increment in time)
    time = np.arange(0, interval, dt)    # time vector over interval

	# generate the sin signal
    x = amplitude*np.sin(2*math.pi*frequency*time+phase)

    return time, x

if __name__ == "__main__":

    Fs = 100.0          # sampling rate
    interval = 1.0      # set up to one full second

    # generate the user-defined sin function
    time, x = generateSin(Fs, interval)
    # use np.random.randn() for randomization
	# we can owverwrie the default values
    # frequency =  8.0                     # frequency of the signal
    # amplitude =  3.0                     # amplitude of the signal
    # phase = 1.0                          # phase of the signal
    # time, x = generateSin(Fs, interval, frequency, amplitude, phase)

    # plot the signal in time domain
    plotTime(x, time)
    # plot the signal in frequency domain
    plotSpectrum(x, Fs, type = 'FFT')

    # compute the spectrum with your own DFT
    # you can use cmath.exp() for complex exponentials
    # plotSpectrum(x, Fs, type = 'your DFT name')

    # confirm DFT/IDFT correctness by checking if x == IDFT(DFT(x))
    # Note: you should also numerically check if the
    # signal energy in time and frequency domains is the same

    # generate randomized multi-tone signals
    # plot them in both time and frequency domain

    plt.show()
