import pyaudio
import numpy as np
import multiprocessing
import time
import pika
import matplotlib.pyplot as plt
import matplotlib
import json
import datetime
import os
import h5py
import sys
import signal
import logging
from pydub import AudioSegment
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.signal import bilinear
from scipy.signal import lfilter
from optparse import OptionParser
matplotlib.use('TkAgg')


# Constants
CHUNK = 2**14  # Number of audio samples per chunk
RATE = 44100  # Sample rate (44.1 kHz)
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 2  # Mono audio
DURATION = 10  # Duration for waterfall display in seconds
CALIBRATION_FACTOR = 0.0177

# Initialize message body to be sent to office NUC with RabbitMQ
message_body = {}

# Compute the hamming window
window = np.hamming(CHUNK)

# Data logging
LOG_FILE_NAME = "log"
GROUP_NAME_CH1 = "channel_1"
GROUP_NAME_CH2 = "channel_2"
sample_counter = 0
spectra_mp3_buffer_ch1 = []
spectra_mp3_buffer_ch2 = []
start_time = time.monotonic()

# Initialize RabbitMQ connection
credentials = pika.PlainCredentials('nuctwo', 'nuctwo')
connection = pika.BlockingConnection(
    pika.ConnectionParameters("10.8.5.157", credentials=credentials) #10.8.5.157
)
channel = connection.channel()
channel.exchange_declare(exchange='log',
                         exchange_type='fanout')

# Initialize the figure with 4 subplots: Frequency plot for each channel + 2 Waterfall plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))

# Set fixed axis limits for the frequency plots at the start
ax1.set_xlim(0, RATE // 2)  # Set fixed x-axis (frequency) range
ax1.set_ylim(-40, 120)  # Set fixed y-axis (magnitude) range

ax2.set_xlim(0, RATE // 2)  # Set fixed x-axis (frequency) range
ax2.set_ylim(-40, 120)  # Set fixed y-axis (magnitude) range

# Data for the waterfall plots (initially empty)
data_ch1 = np.zeros((RATE // CHUNK * DURATION, CHUNK // 2))  # For left channel
data_ch2 = np.zeros((RATE // CHUNK * DURATION, CHUNK // 2))  # For right channel

# Frequency bins for plotting
freqs = np.fft.fftfreq(CHUNK, 1 / RATE)[0 : CHUNK // 2]

# Waterfall plot settings
positions = [0, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1]
colors = [
    "white", "green", "lightgreen", "yellow", "orange", "red", "darkred", "black",
]
cmap = cm.colors.LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))

# Left Channel Waterfall Plot
im_ch1 = ax3.imshow(
    data_ch1, aspect="auto", origin="lower", norm=LogNorm(vmin=1, vmax=100), cmap=cmap
)
ax3.set_title("Waterfall (Left)")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Time Elapsed")
# Remove the y-axis numbers on the left channel waterfall plot
ax3.set_yticks([])  # Hides the y-axis tick marks and labels
# Right Channel Waterfall Plot
im_ch2 = ax4.imshow(
    data_ch2, aspect="auto", origin="lower", norm=LogNorm(vmin=1, vmax=100), cmap=cmap
)
ax4.set_title("Waterfall (Right)")
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Time Elapsed")
# Remove the y-axis numbers on the left channel waterfall plot
ax4.set_yticks([])  # Hides the y-axis tick marks and labels

plt.colorbar(im_ch1, ax=ax3)
plt.colorbar(im_ch2, ax=ax4)

# Set up logging
logging.basicConfig(
    filename='rabbitmq_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_frequency(ax, channel_data_fft_db, title):
    # Check if the line already exists, otherwise create a new line
    if not ax.lines:
        # Create the plot with initial data
        ax.plot(freqs, channel_data_fft_db, label='Channel Spectrum', alpha=0.7)
        
        # Set the labels and title just once (on the first plot)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("SPL [dBA]")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='upper right')  # Set the legend in a consistent location
    else:
        # Update the existing line with new data
        ax.lines[0].set_ydata(channel_data_fft_db)

# Function to acquire audio data and put it into a queue
def audio_acquisition(data_queue_ch1, data_queue_ch2):

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    try:
        while True:

            # Read raw audio data from the stream
            raw_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            left_channel_data = raw_data[::2]
            right_channel_data = raw_data[1::2]

            data_queue_ch1.put(left_channel_data)  # Put the data in the queue for processing
            data_queue_ch2.put(right_channel_data)

    except Exception as e:
        print(f"Error in audio acquisition: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def audio_calculation(data_queue, b, a):
    audio_data = data_queue.get()  # Get the raw audio data from the queue

    data_no_dc = audio_data - np.mean(audio_data)
    data_filtered = lfilter(b, a, data_no_dc)

    data_fft = np.fft.fft(window * data_filtered)

    # Normalize the FFT by the window length
    data_fft_normalized = np.sqrt(2) * data_fft[:CHUNK // 2] / CHUNK

    calibrated_magnitude_db = 20 * np.log10(np.abs(data_fft_normalized) / CALIBRATION_FACTOR)

    # Calculate and display the instantaneous SPL for the left channel
    rms_time = np.sqrt(np.mean(np.square(data_filtered*window)))  # RMS of the raw signal
    rms_frequency_single = np.sqrt(np.sum(np.square(np.abs(data_fft_normalized)))) # RMS of the fft

    # print("rms_time = ", rms_time) # ENABLE THIS FOR CALIBRATION
    # print("rms_frequency_single = ", rms_frequency_single) # ENABLE THIS FOR CALIBRATION
    # spl_dba = 20 * np.log10(rms_time / CALIBRATION_FACTOR)  # SPL in dBA (20 µPa reference) # ENABLE THIS FOR CALIBRATION
    # spl_dba_freq_single = 20 * np.log10(rms_frequency_single / CALIBRATION_FACTOR)  # SPL in dBA (20 µPa reference)
    # print("spl_dba_time = ", left_spl_dba) # ENABLE THIS FOR CALIBRATION 
    # print("spl_dba_freq_single = ", spl_dba_freq_single)
 
    return calibrated_magnitude_db

def process_data_logging(data_left, data_right, opts):
    global sample_counter

    max_ch1 = np.max(data_left)
    max_ch2 = np.max(data_right)
    spec_and_freq_ch1 = np.vstack((data_left, freqs))
    spec_and_freq_ch2 = np.vstack((data_right, freqs))

    log_message_ch1 = (
            "___lab.py:___ PEAK:"
            + str(int(max_ch1))
            + ", Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
    
    log_message_ch2 = (
            "___lab.py:___ PEAK:"
            + str(int(max_ch2))
            + ", Date:"
            + datetime.datetime.now().strftime("%Y-%m-%d")
            + ", Time:"
            + datetime.datetime.now().strftime("%H:%M:%S")
            + '.' + datetime.datetime.now().strftime("%f")[:3]
        )
    
    # Take a sample at midnight
    now = datetime.datetime.now()
    if (0, 0) == (now.hour, now.minute) and 0 < now.second < 5:
        log_message(log_message_ch1, log_option=f"midnight", group_name=f"channel_1", spectrum=spec_and_freq_ch1)
        log_message(log_message_ch2, log_option=f"midnight", group_name=f"channel_2", spectrum=spec_and_freq_ch2)

        spectra_mp3_buffer_ch1.append(data_left)
        spectra_mp3_buffer_ch2.append(data_right)

        mp3_log_message_ch1 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch1"
        mp3_log_message_ch2 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch2"

        mp3_file_location = os.path.join("log", "midnight")

        save_spectra(spectra_mp3_buffer_ch1, mp3_log_message_ch1, output_directory=mp3_file_location)
        save_spectra(spectra_mp3_buffer_ch2, mp3_log_message_ch2, output_directory=mp3_file_location)

    if opts.samples is not None:
        if sample_counter < opts.samples:
            log_message(log_message_ch1, log_option=f"samples", group_name=f"channel_1", spectrum=spec_and_freq_ch1)
            log_message(log_message_ch2, log_option=f"samples", group_name=f"channel_2", spectrum=spec_and_freq_ch2)

            spectra_mp3_buffer_ch1.append(data_left)
            spectra_mp3_buffer_ch2.append(data_right)

            mp3_log_message_ch1 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch1"
            mp3_log_message_ch2 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch2"

            mp3_file_location = os.path.join("log", "samples")

            save_spectra(spectra_mp3_buffer_ch1, mp3_log_message_ch1, output_directory=mp3_file_location)
            save_spectra(spectra_mp3_buffer_ch2, mp3_log_message_ch2, output_directory=mp3_file_location)

            sample_counter += 1
        else:
            print("Sample logging has stopped")
            plt.close()
            connection.close()
            exit()
    
    elif opts.time is not None:
        if opts.time > 0:
            end_time = time.monotonic()
            elapsed_time = end_time - start_time
            if elapsed_time < opts.time:
                log_message(log_message_ch1, log_option=f"time", group_name=f"channel_1", spectrum=spec_and_freq_ch1)
                log_message(log_message_ch2, log_option=f"time", group_name=f"channel_2", spectrum=spec_and_freq_ch2)

                spectra_mp3_buffer_ch1.append(data_left)
                spectra_mp3_buffer_ch2.append(data_right)

                mp3_log_message_ch1 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch1"
                mp3_log_message_ch2 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch2"

                mp3_file_location = os.path.join("log", "time")

                save_spectra(spectra_mp3_buffer_ch1, mp3_log_message_ch1, output_directory=mp3_file_location)
                save_spectra(spectra_mp3_buffer_ch2, mp3_log_message_ch2, output_directory=mp3_file_location)

                sample_counter += 1
            else:
                # If elapsed time exceeds opts.time, stop logging
                print("Logging time has expired.")
                # Optionally, close any resources or finish up logging. There is still a problem with how I'm closing my program.
                connection.close()
                plt.close()
                sys.exit(1)


    elif opts.cont:
        log_message(log_message_ch1, log_option=f"cont", group_name=f"channel_1", spectrum=spec_and_freq_ch1)
        log_message(log_message_ch2, log_option=f"cont", group_name=f"channel_2", spectrum=spec_and_freq_ch2)

        spectra_mp3_buffer_ch1.append(data_left)
        spectra_mp3_buffer_ch2.append(data_right)

        mp3_log_message_ch1 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch1"
        mp3_log_message_ch2 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch2"

        mp3_file_location = os.path.join("log", "cont")

        save_spectra(spectra_mp3_buffer_ch1, mp3_log_message_ch1, output_directory=mp3_file_location)
        save_spectra(spectra_mp3_buffer_ch2, mp3_log_message_ch2, output_directory=mp3_file_location)

    else:
        if max_ch1 > 82:
            log_message(log_message_ch1, log_option=f"", group_name=f"channel_1", spectrum=spec_and_freq_ch1)
            mp3_log_message = (
            datetime.datetime.now().strftime("%Y-%m-%d")
            + datetime.datetime.now().strftime("%H:%M:%S")
            )

            mp3_file_location = os.path.join("log", "danger")
            spectra_mp3_buffer_ch1.append(data_left)
            mp3_log_message_ch1 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch1"
            save_spectra(spectra_mp3_buffer_ch1, mp3_log_message_ch1, output_directory=mp3_file_location)
            
        if max_ch2 > 82:
            log_message(log_message_ch2, log_option=f"", group_name=f"channel_2", spectrum=spec_and_freq_ch2)
            mp3_log_message = (
            datetime.datetime.now().strftime("%Y-%m-%d")
            + datetime.datetime.now().strftime("%H:%M:%S")
            )

            mp3_file_location = os.path.join("log", "danger")
            spectra_mp3_buffer_ch2.append(data_right)
            mp3_log_message_ch2 = str(datetime.datetime.now().strftime("%Y-%m-%d")) + f"_ch2"
            save_spectra(spectra_mp3_buffer_ch2, mp3_log_message_ch2, output_directory=mp3_file_location)


def log_message(dataset_name, group_name, log_option, spectrum=None):
    """
    Writes a log message in a directory called "samples" within the "log"
    directory. It creates the directory if it doesn't exist already.

    Parameters:
        -   log_file (string): Name of directory containing the sub-directories
            for logging.

        -   dataset_name (string): Name of the dataset that the new
            log entry will be written to.

        -   group_name (string): Name of the group that the new log entry will
            be written to.

        -   spectrum (2D numPy array): Frequency components and Frequencies.

    Returns: void/nothing
    """

    log_file = os.path.join(LOG_FILE_NAME, log_option)

    # Create directory if it doesn't exist
    if not os.path.exists(log_file):
        os.makedirs(log_file)

    # Get current date
    today = datetime.date.today()
    log_file = os.path.join(log_file, f"log_{today}.{log_option}.hdf5")

    if not os.path.isfile(log_file):
        # Create the file if it doesn't exist
        with h5py.File(log_file, "w"):
            pass

    with h5py.File(log_file, "a") as f:
        if group_name not in f:
            # Create the file if it doesn't exist
            f.create_group(group_name)

        # Create dataset for data array if provided
        if spectrum is not None:
            f[group_name].create_dataset(dataset_name, data=spectrum, dtype=np.float32)

def save_spectra(spectra_list, mp3_file_name, output_directory='output_mp3'):
    """
    Save spectra to MP3 files. Each file corresponds to a single chunk of data.
    
    Parameters:
    - spectra_list: list of numpy arrays (spectra) to be saved
    - mp3_file_name: name of the MP3 file to be saved
    - output_directory: directory where MP3 files will be saved
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Initialize buffer
    buffer = AudioSegment.empty()
    
    for spectra in spectra_list:
        # Convert spectra to audio segment
        audio_segment = spectra_to_audio(spectra, RATE)
        
        if audio_segment.duration_seconds > 0:  # Check if audio_segment has valid duration
            buffer += audio_segment
        else:
            print("Warning: Empty audio segment detected.")
    
    # Save the accumulated buffer to an MP3 file
    if buffer.duration_seconds > 0:  # Ensure there's data to save
        output_file = os.path.join(output_directory, mp3_file_name)
        try:
            buffer.export(output_file, format='mp3')
            print(f'Saved {output_file}')
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")
    else:
        print("No data to save.")

def spectra_to_audio(spectra, sample_rate=44100):
    """
    Convert spectra (time-domain samples) to an audio segment.
    
    Parameters:
    - spectra: numpy array of time-domain samples
    - sample_rate: sampling rate of the audio
    
    Returns:
    - audio_segment: pydub AudioSegment
    """
    # Normalize to 16-bit PCM
    spectra = np.int16(spectra / np.max(np.abs(spectra)) * 32767)
    
    # Convert numpy array to bytes
    audio_data = spectra.tobytes()
    
    # Create an AudioSegment
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # 16-bit PCM
        frame_rate=sample_rate,
        channels=1  # Mono
    )
    
    return audio_segment

def process_audio(data_queue_ch1, data_queue_ch2, plot_queue_ch1, plot_queue_ch2, b, a, opts):
    try:
        while True:
            if not data_queue_ch1.empty() and not data_queue_ch2.empty():
                audio_data_ch1 = data_queue_ch1
                audio_data_ch2 = data_queue_ch2

                calibrated_left_magnitude_db = audio_calculation(audio_data_ch1, b, a)
                calibrated_right_magnitude_db = audio_calculation(audio_data_ch2, b, a)

                # Put the audio data into a plot queue for plotting
                plot_queue_ch1.put(calibrated_left_magnitude_db)
                plot_queue_ch2.put(calibrated_right_magnitude_db)

                process_data_egress(calibrated_left_magnitude_db, calibrated_right_magnitude_db)
                process_data_logging(calibrated_left_magnitude_db, calibrated_right_magnitude_db, opts)
            else:
                time.sleep(0.01)  # Sleep for a short period to prevent 100% CPU usage
    except Exception as e:
        logging.error(f"Error connecting to RabbitMQ: {str(e)}")
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print(f"Error in audio processing (ch1): {e}")


def process_data_egress(data_ch1, data_ch2):

    # Preparing message body
    message_body = {
        "calibrated_left_magnitude_db": data_ch1.astype(np.float16).tolist(),
        "calibrated_right_magnitude_db": data_ch2.astype(np.float16).tolist()
    }


    # Convert message_body to JSON string
    message_body_json = json.dumps(message_body)

    # Send to RabbitMQ
    channel.basic_publish(exchange='log', routing_key='', body=message_body_json)


def update(frame, plot_queue_ch1, plot_queue_ch2, data_ch1, data_ch2, im_ch1, im_ch2):
    if not plot_queue_ch1.empty() and not plot_queue_ch2.empty():

        # Retrieve the latest FFT data from the plot queues
        calibrated_left_magnitude_db = plot_queue_ch1.get()
        calibrated_right_magnitude_db = plot_queue_ch2.get()

        # Update frequency plots
        plot_frequency(ax1, calibrated_left_magnitude_db, title="SPL vs Frequency (Left)")
        plot_frequency(ax2, calibrated_right_magnitude_db, title="SPL vs Frequency (Right)")

        # Update the waterfall plot for left channel
        data_ch1[:-1, :] = data_ch1[1:, :]  # Roll data down by one row
        data_ch1[-1, :] = calibrated_left_magnitude_db  # Add new data at the end
        im_ch1.set_data(data_ch1)  # Update the waterfall plot
        im_ch1.set_extent([freqs[0], freqs[-1], 0, DURATION])  # Adjust the extent

        # Update the waterfall plot for right channel
        data_ch2[:-1, :] = data_ch2[1:, :]  # Roll data down by one row
        data_ch2[-1, :] = calibrated_right_magnitude_db  # Add new data at the end
        im_ch2.set_data(data_ch2)  # Update the waterfall plot
        im_ch2.set_extent([freqs[0], freqs[-1], 0, DURATION])  # Adjust the extent

        # Return only the updated images for blitting
        return [im_ch1, im_ch2]

    return []  # Return an empty list if no data available

def A_weighting(fs):
    """
    Design of an A-weighting filter.
    
    Parameters:
    fs (int): Sampling rate (Hz)

    Returns:
    b, a : Filter coefficients for use in scipy.signal.lfilter
    """

    # Precompute constant values
    pi = np.pi
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    
    # Precompute the terms for the filter design
    # Calculate constant terms involving `2 * pi * fX`
    w1 = 2 * pi * f1
    w2 = 2 * pi * f2
    w3 = 2 * pi * f3
    w4 = 2 * pi * f4

    # Define the numerator and denominator for the analog filter
    NUMs = [(w4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]

    DENs = np.polymul([1, w4, w4 ** 2], [1, w1, w1 ** 2])
    DENs = np.polymul(np.polymul(DENs, [1, w3]), [1, w2])

    # Use the bilinear transformation to convert the analog filter to a digital filter
    b, a = bilinear(NUMs, DENs, fs)

    return b, a

def parse_args():
    parser = OptionParser()
    parser.add_option(
        "-s",
        "--samples",
        help="Logs N-samples",
        metavar="N",
        type="int"
    )
    parser.add_option(
        "-t",
        "--time",
        help="Logs samples within a time period",
        metavar="TIME",
        type="int",
    )

    parser.add_option(
        "-c",
        "--cont",
        help="Logs samples continuously",
        default=False,
        action="store_true",
    )

    (opts, args) = parser.parse_args()

    return opts, args

# Function to handle Ctrl-C gracefully
def handle_shutdown(signal, frame):
    print("\nGracefully shutting down...")
    
    # Close the RabbitMQ connection
    try:
        connection.close()
    except Exception as e:
        print(f"Error closing RabbitMQ connection: {e}")
    
    # Close the Matplotlib plot
    try:
        plt.close()
    except Exception as e:
        print(f"Error closing Matplotlib plot: {e}")

    # If you're using PyAudio, terminate it cleanly
    try:
        p.terminate()  # Assuming `p` is your PyAudio instance
    except Exception as e:
        print(f"Error closing PyAudio: {e}")
    
    # Exit the program
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, handle_shutdown)

# Main function to run the process and create the animation
if __name__ == "__main__":

    try:
        opts, args = parse_args()

        # Compute A-filter weights
        b, a = A_weighting(RATE)

        # Create multiprocessing Queues for sharing data between processes
        data_queue_ch1 = multiprocessing.Queue()
        data_queue_ch2 = multiprocessing.Queue()
        plot_queue_ch1 = multiprocessing.Queue()
        plot_queue_ch2 = multiprocessing.Queue()

        # Create the processes for audio acquisition, audio processing, and plotting
        acquisition_process = multiprocessing.Process(target=audio_acquisition, args=(data_queue_ch1, data_queue_ch2))
        audio_process = multiprocessing.Process(target=process_audio, args=(data_queue_ch1, data_queue_ch2, plot_queue_ch1, plot_queue_ch2, b, a, opts))

        # Start the processes
        acquisition_process.start()
        audio_process.start()

        # Create the animation
        ani = FuncAnimation(
            fig, update, fargs=(plot_queue_ch1, plot_queue_ch2, data_ch1, data_ch2, im_ch1, im_ch2),
            interval=50, blit=True
        )

        # Show the plot
        plt.show()

        # Wait for the processes to finish (in this case, this will run indefinitely)
        acquisition_process.join()
        audio_process.join()

    except Exception as e:
        print(f"Error in main loop: {e}")
        sys.edit(1)
