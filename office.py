import pika
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import time
import threading
import multiprocessing

# Constants
CHUNK = 2**14  # Number of audio samples per chunk
RATE = 44100  # Sample rate (44.1 kHz)
CALIBRATION_FACTOR = 0.0238
DURATION = 10  # Duration for waterfall display in seconds

# Prepare the figure and axes
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))

# Set frequency plot axes limits
ax1.set_xlim(0, RATE // 2)  # x-axis: frequency range
ax1.set_ylim(-40, 120)  # y-axis: magnitude range
ax2.set_xlim(0, RATE // 2)  # x-axis: frequency range
ax2.set_ylim(-40, 120)  # y-axis: magnitude range

# Data for the waterfall plots (initialized to zero)
data_ch1 = np.zeros((RATE // CHUNK * DURATION, CHUNK // 2))  # Left channel
data_ch2 = np.zeros((RATE // CHUNK * DURATION, CHUNK // 2))  # Right channel

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
ax3.set_yticks([])  # Remove y-axis ticks

# Right Channel Waterfall Plot
im_ch2 = ax4.imshow(
    data_ch2, aspect="auto", origin="lower", norm=LogNorm(vmin=1, vmax=100), cmap=cmap
)
ax4.set_title("Right Channel Waterfall")
ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Time Elapsed")
ax4.set_yticks([])  # Remove y-axis ticks

plt.colorbar(im_ch1, ax=ax3)
plt.colorbar(im_ch2, ax=ax4)

# Function to update frequency plot
def plot_frequency(ax, calibrated_magnitude_db, title):
    if not ax.lines:
        ax.plot(freqs, calibrated_magnitude_db, label='Channel Spectrum', alpha=0.7)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc='upper right')
    else:
        ax.lines[0].set_ydata(calibrated_magnitude_db)

# Callback for Left Channel (callback_ch1)
def callback_ch1(ch, method, properties, body):
    # Deserialize the JSON message body
    message_body = json.loads(body)
    print("In callback ch1")

    # Process left channel message
    left_magnitude_exists = "calibrated_left_magnitude_db" in message_body
    if left_magnitude_exists:
        calibrated_left_magnitude_db = np.array(message_body["calibrated_left_magnitude_db"])
        plot_queue_ch1.put(calibrated_left_magnitude_db)
    else:
        print("Warning: Missing 'calibrated_left_magnitude_db' in the message.")

    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Callback for Right Channel (callback_ch2)
def callback_ch2(ch, method, properties, body):
    # Deserialize the JSON message body
    message_body = json.loads(body)
    print("In callback ch2")

    # Process right channel message
    right_magnitude_exists = "calibrated_right_magnitude_db" in message_body
    if right_magnitude_exists:
        calibrated_right_magnitude_db = np.array(message_body["calibrated_right_magnitude_db"])
        plot_queue_ch2.put(calibrated_right_magnitude_db)
    else:
        print("Warning: Missing 'calibrated_right_magnitude_db' in the message.")

    # Acknowledge the message
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Start the consumers for both channels
def start_consumer_ch1():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='log_ch1', exchange_type='fanout')

    result_ch1 = channel.queue_declare(queue='', exclusive=True)
    queue_ch1 = result_ch1.method.queue
    channel.queue_bind(exchange='log_ch1', queue=queue_ch1)

    print(' [*] Waiting for left channel logs. To exit press CTRL+C')

    channel.basic_consume(queue=queue_ch1, on_message_callback=callback_ch1)
    channel.start_consuming()

def start_consumer_ch2():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.exchange_declare(exchange='log_ch2', exchange_type='fanout')

    result_ch2 = channel.queue_declare(queue='', exclusive=True)
    queue_ch2 = result_ch2.method.queue
    channel.queue_bind(exchange='log_ch2', queue=queue_ch2)

    print(' [*] Waiting for right channel logs. To exit press CTRL+C')

    channel.basic_consume(queue=queue_ch2, on_message_callback=callback_ch2)
    channel.start_consuming()

# Update function for real-time plotting
def update(frame, plot_queue_ch1, plot_queue_ch2, data_ch1, data_ch2, im_ch1, im_ch2):
    if not plot_queue_ch1.empty() and not plot_queue_ch2.empty():

        # Retrieve the latest FFT data from the plot queues
        calibrated_left_magnitude_db = plot_queue_ch1.get()
        calibrated_right_magnitude_db = plot_queue_ch2.get()

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

# Main function
if __name__ == "__main__":
    # Queues to store FFT data for plotting
    plot_queue_ch1 = multiprocessing.Queue()
    plot_queue_ch2 = multiprocessing.Queue()

    # Start the consumer processes for both channels
    consumer_process_ch1 = multiprocessing.Process(target=start_consumer_ch1)
    consumer_process_ch2 = multiprocessing.Process(target=start_consumer_ch2)
    consumer_process_ch1.start()
    consumer_process_ch2.start()

    # Start the animation for real-time plotting
    ani = FuncAnimation(
        fig, update, fargs=(plot_queue_ch1, plot_queue_ch2, data_ch1, data_ch2, im_ch1, im_ch2),
        interval=50, blit=True
    )

    plt.show()

    consumer_process_ch1.join()
    consumer_process_ch2.join()
