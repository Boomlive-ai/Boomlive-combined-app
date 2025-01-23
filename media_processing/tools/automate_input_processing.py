from flask import jsonify
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from media_processing.video_processing import process_video_file
from media_processing.audio_processing import process_audio_file
from media_processing.image_processing import extract_text_from_image
from media_processing.url_processing import scrape_url
import requests
from bs4 import BeautifulSoup
# Helper function to check for allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'mp3', 'wav', 'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Function to detect content type and call the respective processing function

def detect_and_process_file(file, file_path):
    extension = file.filename.rsplit('.', 1)[1].lower()

    try:
        if extension in {'mp4', 'avi', 'mov', 'mkv', 'flv'}:
            # Process video
            video_clip = VideoFileClip(file_path)
            duration = video_clip.duration
            transcript = process_video_file(file_path).text
            video_clip.close()
            return jsonify({"filename": file.filename, "duration": duration, "text": transcript}), 200

        elif extension in {'mp3', 'wav'}:
            # Process audio
            transcript = process_audio_file(file_path)
            return jsonify({"filename": file.filename, "text": transcript}), 200

        elif extension in {'jpg', 'jpeg', 'png', 'webp'}:
            # Process image
            text_from_image = extract_text_from_image(file_path)
            return jsonify({"filename": file.filename, "text": text_from_image}), 200

        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


def detect_and_process_json(data):
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    if 'text' in data:
        return jsonify({"text": data['text']}), 200

    elif 'url' in data:
        return scrape_url(data['url'])

    else:
        return jsonify({"error": "No recognizable JSON input"}), 400