import cv2
import numpy as np
import os
import pyaudio
import wave
import subprocess

# Path to FFmpeg (Ensure this is the correct path to ffmpeg.exe)
FFMPEG_PATH = r"C:\Users\Lenovo\Downloads\ffmpeg\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# Path to the folder containing images
image_folder = r"C:\Users\Lenovo\Downloads\images"
video_name = r"C:\Users\Lenovo\Downloads\output_video.avi"
audio_file = r"C:\Users\Lenovo\Downloads\recorded_audio.wav"
output_video_with_audio = r"C:\Users\Lenovo\Downloads\final_video_with_audio.mp4"

fps = 30  # Frames per second
duration_per_image = 2  # Duration of each image in seconds
num_frames_per_image = fps * duration_per_image


# Function to record your voice
def record_audio(output_file, duration=20):
    p = pyaudio.PyAudio()
    rate = 44100  # Sample rate (Hz)
    channels = 1  # Mono audio
    frames_per_buffer = 1024
    format = pyaudio.paInt16  # 16-bit resolution

    stream = p.open(
        format=format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )

    print("Recording audio...")
    frames = []

    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))


# Record audio
record_audio(audio_file, duration=20)

# Check if the audio file exists
if not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' does not exist.")
    exit()

# Get all images from the folder
images = [
    img for img in os.listdir(image_folder) if img.lower().endswith((".jpg", ".png"))
]

# Check if there are images
if not images:
    print("No images found in the specified folder.")
    exit()

print("Images found:", images)

# Create a VideoWriter object
first_image_path = os.path.join(image_folder, images[0])
print(f"First image path: {first_image_path}")

first_image = cv2.imread(first_image_path)

if first_image is None:
    print(f"Error loading image: {first_image_path}")
    exit()

height, width, layers = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through each image and create video
for image_name in images:
    image_path = os.path.join(image_folder, image_name)
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    for i in range(num_frames_per_image):
        zoom_factor = 1 + (i / num_frames_per_image) * 0.5

        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)

        resized_image = cv2.resize(image, (new_width, new_height))

        canvas = np.zeros((height, width, layers), dtype=np.uint8)

        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2

        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized_image
        )

        video.write(canvas)

video.release()

print(f"Video '{video_name}' has been created successfully.")

# Now, add the recorded audio to the video using FFmpeg
try:
    if not os.path.exists(video_name):
        print(f"Error: Video file '{video_name}' does not exist.")
        exit()
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist.")
        exit()

    # Run FFmpeg with the correct executable path
    ffmpeg_cmd = [
        FFMPEG_PATH,  # Use the full path to ffmpeg.exe
        "-i",
        video_name,
        "-i",
        audio_file,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        "-y",
        output_video_with_audio,
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Video with audio has been saved as '{output_video_with_audio}'")
except subprocess.CalledProcessError as e:
    print(f"Error adding audio: {e}")
except PermissionError:
    print("Permission denied. Try running the script as Administrator.")
