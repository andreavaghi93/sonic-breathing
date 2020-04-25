import numpy as np
import pywt
import peakutils
import matplotlib
from scipy.signal import butter, lfilter
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

def perform_fft_analysis(time_series, buffer_size, peak_treshold):
    # length of the signal and data
    n = len(time_series)

    # normalizing to discard DC component
    mean = sum(time_series) / n
    time_series = time_series - mean

    # zero padding
    pow_2 = 1 << (n-1).bit_length()
    time_series_padded = np.pad(time_series, (0, pow_2 - n), 'constant')
    n_pad = len(time_series_padded)
    fs = buffer_size  # sampling rate
    k = np.arange(n_pad)
    t = n_pad / fs
    frq = k / t  # two sides frequency range
    frq = frq[range(int(n_pad / 2))]  # one side frequency range

    # computing FFT
    fft_x = np.fft.fft(time_series_padded) / n_pad  # (normalized)
    fft_x = fft_x[range(int(n_pad / 2))]
    fft_abs = abs(fft_x)
    print(max(fft_abs))

    # finding peaks
    indexes = peakutils.indexes(fft_abs, thres=peak_treshold, min_dist=20)
    if len(indexes) > 0:
        detected_freq = max(frq[indexes])
    else:
        detected_freq = 0
    if max(fft_abs) > 7:
        return fft_abs, detected_freq
    else:
        return fft_abs, 0



def perform_wavelet_analysis(time_series, central_freq, sampling_rate, peak_tresh):
    mean = sum(time_series) / len(time_series)
    time_series = time_series - mean
    ts = 1 / sampling_rate
    central_scale = sampling_rate / central_freq
    wav = pywt.ContinuousWavelet('cmor2-1.0')
    f_bound_high = central_scale / 2
    f_bound_low = central_scale / 2
    bands = 64
    widths = np.linspace(central_scale - f_bound_high, central_scale + f_bound_low, bands)
    # (IMPORTANTE) devo trovare la formula che lega direttamente la frequenza rilevata dalla fft coi parametri da settare per le wavelets
    # in pratica ho capito che a bassissime frequenze non ho bisogno di un numero di bande superiore a 50
    # oltretutto a basse frequenze l'analisi wavelet è molto più lenta
    # un altro parametro molto importante è la larghezza della banda di analisi
    # il limite inferiore (in frequenza) deve essere aumentato
    # 6 Hz -> bands = 100, f_bound_high = central_scale / 20, f_bound_low = central_scale / 80
    # 0.15 Hz -> bands = 20, f_bound_high = central_scale / 3, f_bound_low = central_scale / 1
    cwtmatr, freqs = pywt.cwt(time_series, widths, wav, ts)
    cwt_freq_peaks = []
    times = 0;
    for time_stamp in cwtmatr.T: # [int(len(cwtmatr.T)/3):int(len(cwtmatr.T)*2/3)]:
        abs_ts = abs(time_stamp)
        max_value = max(abs_ts)
        indexes = peakutils.indexes(abs_ts, thres=peak_tresh, min_dist=20)
        for i in indexes:
            cwt_freq_peaks.append(freqs[i])
        times = times+1
    #print("Peaks: " + str(cwt_freq_peaks))
    #print("Freqs: " + str(freqs))
    freq_det_cwt = np.mean(cwt_freq_peaks)
    return cwtmatr, freq_det_cwt


