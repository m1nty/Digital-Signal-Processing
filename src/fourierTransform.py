import matplotlib.pyplot as plt
import numpy as np
import cmath, math
from scipy import signal


def plotSpectrum(x, Fs, type='FFT'):
    n = len(x)  # length of the signal
    df = Fs / n  # frequency increment (width of freq bin)

    # compute Fourier transform, its magnitude and normalize it before plotting
    if type == 'FFT':
        Xfreq = np.fft.fft(x)
        XMag = abs(Xfreq) / n
    else:
        XMag = DFT_plotSpectrum(x)

    # Note: because x is real, we keep only the positive half of the spectrum
    # Note also: half of the energy is in the negative half (not plotted)
    XMag = XMag[0:int(n / 2)]

    # freq vector up to Nyquist freq (half of the sample rate)
    freq = np.arange(0, Fs / 2, df)

    fig, ax = plt.subplots()
    ax.plot(freq, XMag)
    ax.set(xlabel='FFT Frequency (Hz)', ylabel='FFT Magnitude',
           title='FFT Frequency domain plot')
    # fig.savefig("freq.png")
    plt.show()


def plotTime(x, time):
    fig, ax = plt.subplots()
    ax.plot(time, x)
    ax.set(xlabel='Time (sec)', ylabel='Amplitude',
           title='Time domain plot')
    # fig.savefig("time.png")
    plt.show()


def generateSin(Fs, interval, frequency, amplitude, phase):
    dt = 1.0 / Fs  # sampling period (increment in time)
    time = np.arange(0, interval, dt)  # time vector over interval

    # generate the sin signal
    x = amplitude * np.sin(2 * math.pi * frequency * time + phase)

    return time, x


def DFT_plotSpectrum(x):
    N = len(x)
    result = []
    for m in range(N):
        bins = 0
        for k in range(N):
            bins += x[k] * cmath.exp(-2j * np.pi * (float(k * m) / N))
        result.append(abs(bins) / N)
    return result


def IDFT_real(bins):
    N = len(bins)
    result = []
    for k in range(N):
        x = 0
        for m in range(N):
            x += bins[m] * np.exp(2j * np.pi * ((k * m) / N))
        result.append(x.real / N)
    return result


def DFT(x):
    N = len(x)
    result = []
    for m in range(N):
        bins = 0
        for k in range(N):
            bins += x[k] * np.exp(-2j * np.pi * ((k * m) / N))
        result.append(bins)
    return result


def IDFT(bins):
    N = len(bins)
    result = []
    for k in range(N):
        x = 0
        for m in range(N):
            x += bins[m] * np.exp(2j * np.pi * ((k * m) / N))
        result.append(x / N)
    return result


def randomSignal():
    x = np.zeros(1000)
    for i in range(1000):
        x[i] = np.random.randn()
        if x[i] > 10:
            x[i] = 10
        if x[i] < -10:
            x[i] = -10
    return x


def spectralDensity(nums):
    energy = 0
    for k in nums:
        energy += abs(k ** 2)
    return energy


# *****************************TAKEHOME EXERCISE 1*****************************
def generateSquare(N, numberOfPeriods, dutyCycles):
    # Generate sequence of 1's and -1's depending on duty cycle for a single Period
    low = math.floor((1 - dutyCycles) * N)
    samples = [1] * N
    samples[(N - low):] = [-1] * low

    # Sequence repeats for specified number of periods, if greater than 1
    if numberOfPeriods == 1:
        return samples
    else:
        repeatingSequence = samples.copy()
        for x in range(numberOfPeriods - 1):
            for num in samples:
                repeatingSequence.append(num)
        return repeatingSequence


if __name__ == "__main__":
    Fs = 100.0
    interval = 1.0

    # *****************************In-Lab Part 1*****************************

    # #Task a)
    # time,x = generateSin(Fs, interval, 7.0, 5.0, 0.0)
    # plotTime(x, time)
    # plotSpectrum(x, Fs, type="FFT")
    # plotSpectrum(x, Fs, type="DFT")
    # plotTime(IDFT_real(DFT(x)), time)
    #
    # # Task b)
    # rando = randomSignal()
    # print(spectralDensity(DFT(rando))/1000)
    # print(spectralDensity(IDFT(rando)))
    #
    # #Task c)
    # time, sin1 = generateSin(Fs, interval, 4.0, 100.0, 7.0)
    # time, sin2 = generateSin(Fs, interval, 7.0, 20.0, 4.0)
    # time, sin3 = generateSin(Fs, interval, 10.0, 50.0, 20.0)
    # result = sin1 + sin2 + sin3
    # plotTime(result, time)
    # plotSpectrum(result, Fs, type="FFT")

    # *****************************TAKE HOME EXERCISE 1*****************************
    N = 1000  # number of data points within one period
    numberOfPeriods = 1  # number of periods of square wave
    dutyCycle = 0.7
    squareWaves = generateSquare(N, numberOfPeriods, dutyCycle)  # Generates new square wave

    # Creates evenly distributed partitions depending on number of periods, and data points for each, all within
    # specified range which are assigned from the first 2 arguments of function
    t_samples = np.linspace(0, 1, N * (numberOfPeriods))

    plotTime(squareWaves, t_samples)
    plotSpectrum(squareWaves, Fs, type="FFT")
    plt.show()