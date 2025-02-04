
from moviepy.video.io.VideoFileClip import VideoFileClip
from openai import OpenAI
import cv2
import time
import base64
import os
MODEL="gpt-4o"
os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def process_video_file(video_path, seconds_per_frame=2):
    print(video_path)
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames {total_frames}")
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()
    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
    )
    return transcription
    # return base64Frames, audio_path

# # Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
# base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

# transcription = client.audio.transcriptions.create(
#     model="whisper-1",
#     file=open(audio_path, "rb"),
# )

## Generate a summary with visual and audio