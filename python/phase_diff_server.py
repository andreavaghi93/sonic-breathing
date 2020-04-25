import plot_manager
import buffer_manager
import matplotlib
import numpy as np
import threading
import queue
import peakutils
from pythonosc import dispatcher, osc_server, udp_client
from scipy import signal as sig
matplotlib.use('TkAgg')


# DATA VARIABLES

breath_buffer = []      # 250 sample
bc_length = 250         # length of a breath cycle (in samples, depending on the sampling period)

# SYSTEM STATE VARIABLES

recording = False

# SERVER ADDRESS

server_ip = "127.0.0.1"
server_port = 4000
port_max_msp = 5006

# OSC SERVER


class OSC_server:

    def __init__(self, ip, port, s_disp):
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), s_disp)
        self.server_thread = threading.Thread(target=self.server.serve_forever)

    def start_server(self):
        self.server_thread.start()


def start_cycle(message_address, data):
    global breath_buffer, recording
    breath_buffer = []


# POPULATE THE BUFFER(S)

def collect_breath_sample(message_address, data):
    global breath_buffer

    # COLLECTING BREATH CYCLES
    if recording:
        breath_buffer.append(data)
        if len(breath_buffer) == 250:
            breath_buffer_np = process_breath_cycle(np.array(breath_buffer))
            send_osc_max("/python/cycle", breath_buffer_np)
            phase_diff = find_phase_difference(breath_model, breath_buffer_np)
            print("Index position: " + str(int(phase_diff)))
            send_osc_max("/python/phase_diff", int(phase_diff))
    return


def process_breath_cycle(breath_cycle):
    global bc_length

    breath_cycle = np.array(breath_cycle)
    breath_cycle_norm = np.subtract(breath_cycle, breath_cycle.mean())

    return breath_cycle_norm


def find_phase_difference(model, recorded_cycle):
    cross_corr = sig.correlate(model, recorded_cycle)
    send_osc_max("/python/xcorr", cross_corr)
    print("Length crosscorr: %s" % len(cross_corr))
    indexes = peakutils.indexes(cross_corr, thres=0.9, min_dist=10)
    if len(indexes) > 0:
        return indexes[0]
    else:
        return 0


def load_model(message_address, path):
    global breath_model, recording
    print("Session started!")
    breath_model = np.load(str(path).split("sonic_breathing/")[1])
    send_osc_max("/python/trained_model", breath_model)
    recording = True


# SENDING ONSET SIGNAL TO MAX

def trigger_onset():
    global onset_triggered
    print("SENDING TRIGGER")
    send_osc_max("/python/onset", 1)
    onset_triggered = True


def send_osc_max(address, value):
    client.send_message(address, value)


def plot_large_buffer(message_address, args):
    global large_buffer
    time_series = np.array(large_buffer)
    time_series_flt = buffer_manager.flat_buffer(time_series)
    main_thread_queue.put(lambda: callback_plot_time_series(time_series_flt))


def callback_plot_time_series(time_series):
    plot_manager.plot_time_series(time_series)


def callback_print_test_breath():
    global test_breath_cycles, mean_tbc
    plot_manager.plot_breath_test(test_breath_cycles, mean_tbc)


def from_main_thread_blocking():
    callback = main_thread_queue.get() #blocks until an item is available
    callback()


main_thread_queue = queue.Queue()

server_dispatcher = dispatcher.Dispatcher()
server_dispatcher.map("/max/cycle_sample", collect_breath_sample)
server_dispatcher.map("/max/start_cycle", start_cycle)
server_dispatcher.map("/max/model_path", load_model)


if __name__ == "__main__":

    client = udp_client.SimpleUDPClient(server_ip, port_max_msp)

    # server for receiving OSC messages

    s = OSC_server(server_ip, server_port, server_dispatcher)
    s.start_server()

    print("Server is ready!")



