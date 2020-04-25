import plot_manager
import matplotlib
import numpy as np
import threading
import queue
from pythonosc import dispatcher, osc_server, udp_client
from scipy import signal as sig
matplotlib.use('TkAgg')

# DATA VARIABLES

breath_buffer = []      # 250 sample
bc_length = 250         # length of a breath cycle (in samples, depending on the sampling period)

breath_model = []
model_path = None


# TRAINING VARIABLES
current_training_username = ""
test_breath_cycles = []             # 10 breath cycles to be analized in the training phase
mean_tbc = []                       # mean of the test cycles, used as the model
nof_cycles_model = 10                # number of cycles used to build the model


# SYSTEM STATE VARIABLES

recording = False

# SERVER ADDRESS

server_ip = "127.0.0.1"
server_port = 5000
port_max_msp = 5006

# OSC SERVER


class OSC_server:

    def __init__(self, ip, port, s_disp):
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), s_disp)
        self.server_thread = threading.Thread(target=self.server.serve_forever)

    def start_server(self):
        self.server_thread.start()


# POPULATE THE BUFFER(S)

def collect_breath_sample(message_address, data):

    # COLLECTING BREATH CYCLES
    if recording:
        if len(test_breath_cycles) > 0:
            test_breath_cycles[-1].append(data)
    return


def record_breath_cycle(message_address, data):
    global recording, test_breath_cycles, mean_tbc, bc_length, breath_buffer, nof_cycles_model,\
            breath_model, current_training_username
    len_tb = len(test_breath_cycles)
    if recording:
        if len_tb == 0:
            new_test = []
            test_breath_cycles.append(new_test)
        else:
            test_breath_cycles[-1] = np.asarray(process_breath_cycle(test_breath_cycles[-1]))
            print("Iteration %s:" % str(len_tb))
            for t in test_breath_cycles:
                print(len(t))
            if 0 < len_tb < nof_cycles_model:
                new_test = []
                test_breath_cycles.append(new_test)
            elif len_tb == nof_cycles_model:
                recording = False
                mean_tbc = np.array(test_breath_cycles).mean(axis=0)
                main_thread_queue.put(lambda: callback_print_test_breath())
                np.save("breath_models/bm_" + str(current_training_username), mean_tbc)
                send_osc_max("/python/trained_model", mean_tbc)
                reset_training()
                send_osc_max("/python/end_training", 0)


def reset_training():
    global test_breath_cycles, mean_tbc, current_training_username, recording
    print("Training phase for user %s ended!" % current_training_username)
    recording = False
    test_breath_cycles = []
    current_training_username = ""
    mean_tbc = []


def process_breath_cycle(breath_cycle):
    global bc_length

    breath_cycle = np.array(breath_cycle)
    breath_cycle_norm = np.subtract(breath_cycle, breath_cycle.mean())

    return breath_cycle_norm


def start_training_phase(message_address, username):
    global recording, training_phase, current_training_username
    recording = True
    current_training_username = str(username).replace(" ", "_").lower()
    print("Training phase started!")


def send_osc_max(address, value):
    client.send_message(address, value)


def callback_plot_time_series(time_series):
    plot_manager.plot_time_series(time_series)


def callback_print_test_breath():
    global test_breath_cycles, mean_tbc
    plot_manager.plot_breath_test(test_breath_cycles, mean_tbc)


def from_main_thread_blocking():
    callback = main_thread_queue.get()
    callback()


main_thread_queue = queue.Queue()

server_dispatcher = dispatcher.Dispatcher()
server_dispatcher.map("/max/training_sample", collect_breath_sample)
server_dispatcher.map("/max/start_cycle_training", record_breath_cycle)
server_dispatcher.map("/max/start_training_phase", start_training_phase)


if __name__ == "__main__":

    client = udp_client.SimpleUDPClient(server_ip, port_max_msp)

    # server for receiving OSC messages

    s = OSC_server(server_ip, server_port, server_dispatcher)
    s.start_server()

    print("Server is ready!")



