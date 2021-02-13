import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import signal
from filterDesign import lowPass
import math

def filter_block_processing(audio_data, \
							block_size, \
							audio_Fc, \
							audio_Fs, \
							N_taps):

	# derive filter coefficients
	coeff = lowPass(audio_Fc, audio_Fs, N_taps)

	# we assume the data is stereo as in the audio test file
	filtered_data = np.empty(shape = audio_data.shape)
	# start at the first block (with relative position zero)
	position = 0
	z = signal.lfilter_zi(coeff, 1.0)


	while True:

		# *****************************In-Lab Part 3*****************************


		filtered_data[position:position+block_size, 0], z= signal.lfilter(coeff, 1.0, audio_data[position:position+block_size, 0], zi =z)
		filtered_data[position:position+block_size, 1], z = signal.lfilter(coeff, 1.0, audio_data[position:position+block_size, 1], zi =z)


		# *****************************TAKEHOME EXERCISE 3*****************************

		filtered_data[position:position+block_size, 0]= convolution(audio_data[position:position+block_size, 0], coeff)
		filtered_data[position:position+block_size, 1]= convolution(audio_data[position:position+block_size, 1], coeff)

		position += block_size
		if position > len(audio_data):
			break

	# to properly handle blocks you will need to use
	# the zi argument from lfilter from SciPy
	# explore SciPy, experiment, understand and learn!

	return filtered_data


# *****************************TAKEHOME EXERCISE 3*****************************

def convolution(x, h):
	M = len(x)
	N = len(h)
	y = np.zeros(x.shape[0])
	for n in range(M+N-1): #finite sequence, thus bound ensures all values are covered
		for k in range(N+1):
			if((n-k)>=0 and k<=(N-1) and (n-k)<=(M-1)): #conditions which result in invalid indexing (convolution value of 0)
				y[n] += x[n - k]*h[k] #convolution summation
		if n==len(x)-1: #if block size reached, break from loop
			break
	return y

def filter_single_pass(audio_data, audio_Fc, audio_Fs, N_taps):

	# derive filter coefficients
	coeff = lowPass(audio_Fc, audio_Fs, N_taps)
	# we assume the data is stereo as in the audio test file
	filtered_data = np.empty(shape = audio_data.shape)

	# # filter left channel
	filtered_data[:,0] = convolution(audio_data[:,0],coeff)
	# # filter stereo channel
	filtered_data[:,1] = convolution(audio_data[:,1],coeff)

	return filtered_data

# audio test file from: https://www.videvo.net/royalty-free-music/
if __name__ == "__main__":

	# use use wavfile from scipy.io for handling .wav files
	audio_Fs, audio_data = wavfile.read("../data/audio_test.wav")
	print(' Audio sample rate = {0:f} \
		\n Number of channels = {1:d} \
		\n Numbef of samples = {2:d}' \
		.format(audio_Fs, audio_data.ndim, len(audio_data)))

	# you can control the cutoff frequency and number of taps
	single_pass_data = filter_single_pass(audio_data, \
						audio_Fc = 10e3, \
						audio_Fs = audio_Fs, \
						N_taps = 51)

	# write filtered data back to a .wav file
	wavfile.write("../data/single_pass_filtered.wav", \
	 			audio_Fs, \
				single_pass_data.astype(np.int16))

	# you can control also the block size
	block_processing_data = filter_block_processing(audio_data, \
						block_size = 60, \
						audio_Fc = 10e3, \
						audio_Fs = audio_Fs, \
						N_taps = 51)

	wavfile.write("../data/block_processing_filtered.wav", \
	 			audio_Fs, \
				block_processing_data.astype(np.int16))

	# it is suggested that you add plotting while troubleshooting
	# if you plot in the time domain, select a subset of samples,
	# from a particular channel (or both channels) e.g.,
	# audio_data[start:start+number_of_samples, 0]
	# plt.show()
