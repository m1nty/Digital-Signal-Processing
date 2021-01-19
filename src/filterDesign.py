import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# use generateSin/plotTime from the fourierTransform module
from fourierTransform import generateSin, plotTime

def freqzPlot(coeff, msg):

	# find the frequency response using freqz from SciPy:
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html
    w, h = signal.freqz(coeff)

	# plots the magnitude response where the x axis is normalized in rad/sample
	# Reminder: math.pi rad/sample is actually the Nyquist frequency
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response (' + msg + ')')
    ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [rad/sample]')

	# uncomment the lines below if you wish to inspect the phase response
	# Note: as important as the phase response is, it is not critical
	# at this stage because we expect a linear phase in the passband

    # ax2 = ax1.twinx()
    # angles = np.unwrap(np.angle(h))
    # ax2.plot(w, angles, 'g')
    # ax2.set_ylabel('Angle (radians)', color='g')

def filterSin(Fs, Fc, coeff):

    # we can control the frequency relative to the filter cutoff
	time, x = generateSin(Fs, interval = 1.0, frequency = Fc * 0.4)
	plotTime(x, time)

    # use lfilter from SciPy for FIR filtering:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
	fx = signal.lfilter(coeff, 1.0, x)

    # you should cleary the effects (attenuation, delay) introduced by the filter
	plotTime(fx, time)

if __name__ == "__main__":

	Fs = 100.0           # sampling rate
	Fc = 15.0            # cutoff frequency
	N_taps = 41          # number of taps for the FIR

	# derive filter coefficients using firwin from Scipy:
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
	# second argument is the normalized cutoff frequency, i.e., the
	# cutoff frequency divided by Nyquist frequency (half of sampling rate)
	firwin_coeff = signal.firwin(N_taps, Fc/(Fs/2), window=('hann'))

	# plot the frequency response obtained through freqz
	freqzPlot(firwin_coeff, 'firwin with ' + str(N_taps) + ' taps')

    # implement your own method for finding the coefficients for a low pass filter
    # my_own_coeff = ... provide the following arguments: Fc, Fs and N_taps
    # compare through visual inspection the frequency response against firwin
	# freqzPlot(my_own_coeff, 'my own FIR design with ' + str(N_taps) + ' taps')

	# you can confirm that a single tone has been filtered
	# filterSin(Fs, Fc, firwin_coeff)

	plt.show()
