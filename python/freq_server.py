import analysis_manager
import plot_manager
import buffer_manager
import matplotlib
import numpy as np
import threading
import queue
from pythonosc import dispatcher, osc_server, udp_client
matplotlib.use('TkAgg')

# DATA VARIABLES

small_buffer = []           # 1 SEC BUF
big_buffer = []             # 5 SEC BUF
sampling_frequency = 40     # 1 SEC BUF SIZE - depending on the sampling frequency of the original signal

sample_period_factor = 2
small_buffer_size = sampling_frequency / sample_period_factor
big_buffer_size = 16        # number of small buffer to concatenate to form the big buffer
block_counter = 0           # number of "small buffers" acquired
timerThread = None


# FFT & WAVELETS PARAMETERS

peak_tresh = 0.9            # peakfinder relative threshold
central_freq = 0            # frequency detected from FFT
freq_det_cwt = 0            # frequency detected from CWT
buffer_cwt = None
buffer_fft = None


# SERVER ADDRESS

server_ip = "127.0.0.1"
server_port = 3001
port_max_msp = 5005


# OSC SERVER

class OSCServer:

    def __init__(self, ip, port, s_disp):
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), s_disp)
        self.server_thread = threading.Thread(target=self.server.serve_forever)

    def start_server(self):
        self.server_thread.start()
        print("Server is running, waiting for data..")


# POPULATE THE BUFFER(S)

def collect_data_samples(message_address, data):
    global block_counter, small_buffer, block_counter
    data = data*5
    # SAMPLE COLLECTING AND BUFFER ANALYSIS
    if len(small_buffer) < small_buffer_size:
        small_buffer.append(data)
        if len(small_buffer) == small_buffer_size:
            # pre-filtering
            # small_buffer = buffer_manager.butter_bandpass_filter(small_buffer, 0.5, 6, small_buffer_size, order=5)
            big_buffer.append(small_buffer)
            block_counter = block_counter + 1
            if block_counter > big_buffer_size:  # >= 16 for breathing experiment
                del big_buffer[0]
                main_thread_queue.put(lambda: analyze_buffer())
            big_buffer_flat = buffer_manager.flat_buffer(big_buffer)
            client.send_message("/python/timeseries", big_buffer_flat)
            small_buffer = []
    return


# ANALYZE THE BUFFER

def analyze_buffer():
    global small_buffer, big_buffer, small_buffer_size, central_freq, buffer_fft,\
            buffer_cwt, freq_det_cwt
    time_series = np.array(big_buffer)
    time_series_flt = buffer_manager.flat_buffer(time_series)
    buffer_fft, central_freq = analysis_manager.perform_fft_analysis(time_series_flt,
                                                small_buffer_size * sample_period_factor, peak_tresh)
    print('FFT Peaks are: %s' % central_freq)
    if central_freq > 0:
        buffer_cwt, freq_det_cwt = analysis_manager.perform_wavelet_analysis(time_series_flt,
                                                central_freq, small_buffer_size * sample_period_factor, peak_tresh)
        print('CWT Peaks are: %s' % freq_det_cwt)
        send_osc_max("/python/freq", freq_det_cwt)
    else:
        send_osc_max("/python/freq", 0)
    send_osc_max("/python/fft", buffer_fft)


def send_osc_max(address, value):
    client.send_message(address, value)


def set_peakfinder_thresh(message_address, threshold):
    global peak_tresh
    peak_tresh = threshold


def plot_big_buffer(message_address, args):
    global big_buffer
    time_series = np.array(big_buffer)
    time_series_flt = buffer_manager.flat_buffer(time_series)
    main_thread_queue.put(lambda: callback_plot_time_series(time_series_flt))


def fft_analysis(message_address, args):
    main_thread_queue.put(lambda: callback_fft())


def wavelet_analysis(message_address, args):
    main_thread_queue.put(lambda: callback_wavelets())


def callback_fft():
    global buffer_fft
    plot_manager.plot_fft(buffer_fft)


def callback_wavelets():
    global buffer_cwt
    plot_manager.plot_wavelet(buffer_cwt)


def callback_plot_time_series(time_series):
    plot_manager.plot_time_series(time_series)


def from_main_thread_blocking():
    callback = main_thread_queue.get()
    callback()


if __name__ == "__main__":

    main_thread_queue = queue.Queue()
    server_dispatcher = dispatcher.Dispatcher()
    server_dispatcher.map("/max/signal", collect_data_samples)
    server_dispatcher.map("/max/timeplot", plot_big_buffer)
    server_dispatcher.map("/max/wavelets", wavelet_analysis)
    server_dispatcher.map("/max/fft", fft_analysis)
    server_dispatcher.map("/max/peakfinder_fft", set_peakfinder_thresh)

    client = udp_client.SimpleUDPClient(server_ip, port_max_msp)

    s = OSCServer(server_ip, server_port, server_dispatcher)
    s.start_server()

    while True:
        from_main_thread_blocking()


