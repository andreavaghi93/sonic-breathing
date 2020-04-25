import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def plot_time_series(time_series):
    plt.figure(2)
    plt.plot(time_series, 'r')
    plt.xlabel('Time Series')
    plt.ylabel('Amp')
    plt.show()


def plot_fft(fft):
    plt.figure(1)
    plt.plot(fft)
    plt.show()


def plot_wavelet(cwtmatr):
    plt.figure(1)
    plt.imshow(abs(cwtmatr), extent=[0, len(cwtmatr.T), 0, 30], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()


def plot_breath_test(test_bc, mean_tbc):
    plt.figure(1)
    i = 0
    for test in test_bc:
        i = i + 1
        plt.subplot(len(test_bc)+1, 1, i, ylabel="Amp")
        plt.plot(test)
    plt.subplot(len(test_bc)+1, 1, len(test_bc)+1, ylabel="Amp")
    plt.plot(mean_tbc, 'r')
    plt.show()

