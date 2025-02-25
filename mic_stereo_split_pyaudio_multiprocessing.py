import pyaudio
import numpy as np
import multiprocessing
import time
import pika
import matplotlib.pyplot as plt
import matplotlib
import json
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from matplotlib import cm
from scipy.signal import bilinear
from scipy.signal import lfilter
matplotlib.use('TkAgg')


# Constants
CHUNK = 2**14  # Number of audio samples per chunk
RATE = 44100  # Sample rate (44.1 kHz)
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 2  # Mono audio
DURATION = 10  # Duration for waterfall display in seconds
CALIBRATION_FACTOR = 0.0238

# Initialize message body to be sent to office NUC with RabbitMQ
message_body = {}

# Compute the hamming window
window = np.hamming(CHUNK)

# Initialize RabbitMQ connection
connection = pika.BlockingConnection(
    pika.ConnectionParameters("localhost")
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
ax3.set_title("Left Channel Waterfall")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Time Elapsed")
# Remove the y-axis numbers on the left channel waterfall plot
ax3.set_yticks([])  # Hides the y-axis tick marks and labels
# Right Channel Waterfall Plot
im_ch2 = ax4.imshow(
    data_ch2, aspect="auto", origin="lower", norm=LogNorm(vmin=1, vmax=100), cmap=cmap
)
ax4.set_title("Right Channel Waterfall")
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Time Elapsed")
# Remove the y-axis numbers on the left channel waterfall plot
ax4.set_yticks([])  # Hides the y-axis tick marks and labels

plt.colorbar(im_ch1, ax=ax3)
plt.colorbar(im_ch2, ax=ax4)

def plot_frequency(ax, channel_data_fft_db, title):
    # Check if the line already exists, otherwise create a new line
    if not ax.lines:
        # Create the plot with initial data
        ax.plot(freqs, channel_data_fft_db, label='Channel Spectrum', alpha=0.7)
        
        # Set the labels and title just once (on the first plot)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")
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
    data_fft_normalized = data_fft / CHUNK

    calibrated_magnitude_db = 20 * np.log10(np.abs(data_fft_normalized[:CHUNK // 2]) / CALIBRATION_FACTOR)

    # Calculate and display the instantaneous SPL for the left channel
    rms = np.sqrt(np.mean(np.square(np.abs(calibrated_magnitude_db))))  # RMS of the raw signal
    spl_dba = 20 * np.log10(rms / CALIBRATION_FACTOR)  # SPL in dBA (20 µPa reference)

    return spl_dba, calibrated_magnitude_db

    # Print instantaneous SPL
    # print("left_spl_dba = ", rms_dba)

def process_audio(data_queue_ch1, data_queue_ch2, plot_queue_ch1, plot_queue_ch2, b, a):
    try:
        while True:
            if not data_queue_ch1.empty() and not data_queue_ch2.empty():
                audio_data_ch1 = data_queue_ch1
                audio_data_ch2 = data_queue_ch2

                left_spl_dba, calibrated_left_magnitude_db = audio_calculation(audio_data_ch1, b, a)
                right_spl_dba, calibrated_right_magnitude_db = audio_calculation(audio_data_ch2, b, a)

                # Print instantaneous SPL
                # print("left_spl_dba = ", left_spl_dba)
                # print("right_spl_dba = ", right_spl_dba)

                # Put the audio data into a plot queue for plotting
                plot_queue_ch1.put(calibrated_left_magnitude_db)
                plot_queue_ch2.put(calibrated_right_magnitude_db)

                process_data_egress(calibrated_left_magnitude_db, calibrated_right_magnitude_db)
            else:
                time.sleep(0.01)  # Sleep for a short period to prevent 100% CPU usage
    except Exception as e:
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

        # # Calculate and display the instantaneous SPL for the left channel
        # left_rms = np.sqrt(np.mean(np.square(np.abs(left_channel_filtered))))  # RMS of the raw signal
        # left_spl_dba = 20 * np.log10(left_rms / (20e-6 * CALIBRATION_FACTOR))  # SPL in dBA (20 µPa reference)

        # # Calculate and display the instantaneous SPL for the right channel
        # right_rms = np.sqrt(np.mean(np.square(np.abs(right_channel_filtered))))  # RMS of the raw signal
        # right_spl_dba = 20 * np.log10(right_rms / (20e-6 * CALIBRATION_FACTOR))  # SPL in dBA (20 µPa reference)

        # # Remove previous SPL text (only if it exists)
        # if ax1.texts:
        #     ax1.texts[-1].remove()
        # if ax2.texts:
        #     ax2.texts[-1].remove()

        # # Display SPL on the plot
        # ax1.text(0.95, 0.9, f"SPL (Left): {left_spl_dba:.2f} dBA", transform=ax1.transAxes, fontsize=12, color='black', ha='right')
        # ax2.text(0.95, 0.9, f"SPL (Right): {right_spl_dba:.2f} dBA", transform=ax2.transAxes, fontsize=12, color='black', ha='right')

        # Update frequency plots
        plot_frequency(ax1, calibrated_left_magnitude_db, title="Left Channel")
        plot_frequency(ax2, calibrated_right_magnitude_db, title="Right Channel")

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

# Main function to run the process and create the animation
if __name__ == "__main__":

    # Compute A-filter weights
    b, a = A_weighting(RATE)

    # Create multiprocessing Queues for sharing data between processes
    data_queue_ch1 = multiprocessing.Queue()  # For audio acquisition and processing
    data_queue_ch2 = multiprocessing.Queue()
    plot_queue_ch1 = multiprocessing.Queue()  # For audio data to plot
    plot_queue_ch2 = multiprocessing.Queue()  # For audio data to plot

    # Create the processes for audio acquisition, audio processing, and plotting
    acquisition_process = multiprocessing.Process(target=audio_acquisition, args=(data_queue_ch1,data_queue_ch2))
    audio_process = multiprocessing.Process(target=process_audio, args=(data_queue_ch1, data_queue_ch2, plot_queue_ch1, plot_queue_ch2, b, a))
    # data_egress_process = multiprocessing.Process(target=process_data_egress, args=(plot_queue_ch1, plot_queue_ch2))

    # Start the processes
    acquisition_process.start()
    audio_process.start()
    # data_egress_process.start()

    # Create the animation
    ani = FuncAnimation(
        fig, update, fargs=(plot_queue_ch1, plot_queue_ch2, data_ch1, data_ch2, im_ch1, im_ch2),
        interval=50, blit=True  # Set blit to True to optimize performance
    )

    # Show the plot
    plt.show()

    # Wait for the processes to finish (in this case, this will run indefinitely)
    acquisition_process.join()
    audio_process.join()
    # data_egress_process.join()
