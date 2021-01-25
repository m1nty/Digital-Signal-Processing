import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

# use generateSin/plotTime from the fourierTransform module
from fourierTransform import generateSin, plotTime, plotSpectrum

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


# *****************************In-Lab Part 2*****************************

def lowPass(fc, fs, taps):
	nc = fc/(fs/2)
	h = [0]*taps
	for i in range(taps):
		if(i ==(taps-1)/2):
			h[i] = nc
		else:
			h[i] = nc*( (np.sin(np.pi*nc*(i - (taps-1)/2)))/(np.pi*nc*(i - (taps-1)/2)))
		h[i] = h[i] * (np.sin((i*np.pi)/taps)**2)
	return h

def multiToneFilter(Fs, Fc, coeff):
	interval = 1.0
	# we can control the frequency relative to the filter cutoff
	time, sin1 = generateSin(Fs, interval, 1.0, 100.0, 7.0)
	time, sin2 = generateSin(Fs, interval, 10.0, 20.0, 4.0)
	time, sin3 = generateSin(Fs, interval, 30.0, 50.0, 20.0)
	result = sin1 + sin2 + sin3

	plotTime(result, time)
	plotSpectrum(result, Fs, type="FFT")

	# use lfilter from SciPy for FIR filtering:
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
	fx = signal.lfilter(coeff, 1.0, result)

	# you should cleary the effects (attenuation, delay) introduced by the filter
	plotTime(fx, time)
	plotSpectrum(fx, Fs, type="FFT")
	return result


if __name__ == "__main__":

	Fs = 100.0           # sampling rate
	Fc = 15.0            # cutoff frequency
	N_taps = 41          # number of taps for the FIR

	# derive filter coefficients using firwin from Scipy:
	# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
	# second argument is the normalized cutoff frequency, i.e., the
	# cutoff frequency divided by Nyquist frequency (half of sampling rate)
	# firwin_coeff = signal.firwin(N_taps, Fc/(Fs/2), window=('hann'))
	# plot the frequency response obtained through freqz
	# freqzPlot(firwin_coeff, 'firwin with ' + str(N_taps) + ' taps')
	# freqzPlot(lowPass(Fc, Fs, N_taps), 'lowPass with ' + str(N_taps) + ' taps')

    # implement your own method for finding the coefficients for a low pass filter
    # my_own_coeff = ... provide the following arguments: Fc, Fs and N_taps
    # compare through visual inspection the frequency response against firwin
	# freqzPlot(my_own_coeff, 'my own FIR design with ' + str(N_taps) + ' taps')

	# you can confirm that a single tone has been filtered
	# filterSin(Fs, Fc, firwin_coeff)

	# *****************************PART 2*****************************
	# freqzPlot(lowPass(Fc, Fs, N_taps), 'lowPass with ' + str(N_taps) + ' taps')
	# multiToneFilter(Fs, Fc, lowPass(Fc, Fs, N_taps))



	# *****************************TAKE HOME EXERCISE 2*****************************

	# Frequencies specified for band pass paramaters eliminate the middle tone signal effectivley
	bandPassCoeff = signal.firwin(N_taps, [0.1,0.5], window=('hann'))
	multiToneFilter(Fs, Fc, bandPassCoeff)


	plt.show()
