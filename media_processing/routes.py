from flask import Flask, Blueprint, request, jsonify
from media_processing.video_processing import process_video_file
from media_processing.audio_processing import process_audio_file
from media_processing.image_processing import extract_text_from_image
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from media_processing.tools.automate_input_processing import detect_and_process_file, detect_and_process_json
from bs4 import BeautifulSoup
import requests
from werkzeug.utils import secure_filename
media_processing_bp = Blueprint('media_processing', __name__)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'mp3', 'wav', 'jpg', 'jpeg', 'png', 'webp'}
# Helper function to check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to display API documentation
@media_processing_bp.route('/', methods=['GET'])
def documentation():
    doc = {
        "API Documentation": {
            "upload_video": {
                "description": "Upload a video file and extract its duration and transcript",
                "method": "POST",
                "endpoint": "/upload_video",
                "parameters": {
                    "file": "The video file to upload"
                },
                "example": {
                    "url": "/upload_video"
                },
                "response": {
                    "filename": "Name of the uploaded file",
                    "duration": "Duration of the video in seconds",
                    "path": "Path to the saved file",
                    "transcript": "Transcript extracted from the video"
                }
            },
            "upload_audio": {
                "description": "Upload an audio file and generate a transcript",
                "method": "POST",
                "endpoint": "/upload_audio",
                "parameters": {
                    "file": "The audio file to upload"
                },
                "example": {
                    "url": "/upload_audio"
                },
                "response": {
                    "filename": "Name of the uploaded file",
                    "path": "Path to the saved file",
                    "transcript": "Transcript extracted from the audio"
                }
            },
            "upload_image": {
                "description": "Upload an image file and extract text from it",
                "method": "POST",
                "endpoint": "/upload_image",
                "parameters": {
                    "file": "The image file to upload"
                },
                "example": {
                    "url": "/upload_image"
                },
                "response": {
                    "filename": "Name of the uploaded file",
                    "path": "Path to the saved file",
                    "text": "Text extracted from the image"
                }
            },
            "scrape_url": {
                "description": "Scrape and extract text from a given URL",
                "method": "POST",
                "endpoint": "/scrape_url",
                "parameters": {
                    "url": "The URL to scrape"
                },
                "example": {
                    "url": "/scrape_url"
                },
                "response": {
                    "url": "The scraped URL",
                    "extracted_text": "Text extracted from the URL"
                }
            },
            "process_input": {
                "description": "Process input which can be a file or JSON data",
                "method": "POST",
                "endpoint": "/process_input",
                "parameters": {
                    "file": "A file to upload (optional, depending on the content type)",
                    "json": "JSON data to process (optional)"
                },
                "example": {
                    "url": "/process_input"
                },
                "response": {
                    "message": "Result of processing the input"
                }
            }
        }
    }
    return jsonify(doc)

# Route to upload video
@media_processing_bp.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique file path to store the video in the uploads folder
        file_path = os.path.join('uploads', file.filename)
        
        # Save the uploaded video file locally
        file.save(file_path)
        
        # Process the video (Example: Get duration of the video)
        try:
            video_clip = VideoFileClip(file_path)
            duration = video_clip.duration  # Get the duration in seconds
            print(f"File: {file.filename}, Duration: {duration}")
            transcript = process_video_file(file_path).text
            print(transcript)
            # Ensure the video file is properly closed after processing
            video_clip.close()

            # Return the results (e.g., video duration)
            return jsonify({"filename": file.filename, "duration": duration, "path": file_path, "transcript": transcript}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to process video: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a valid video file."}), 400



# Route to upload and process audio
@media_processing_bp.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique file path to store the audio in the uploads folder
        file_path = os.path.join('uploads', file.filename)
        print(file_path)
        # Save the uploaded audio file locally
        file.save(file_path)
        
        # Process the audio file (Example: Generate transcript)
        try:
            transcript = process_audio_file(file_path)  # Process audio file for transcript
            print(f"Transcript: {transcript}")

            # Return the results (e.g., transcript)
            return jsonify({"filename": file.filename, "path": file_path, "transcript": transcript}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a valid audio file."}), 400


# Route to upload and process image
@media_processing_bp.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique file path to store the image in the uploads folder
        file_path = os.path.join('uploads', file.filename)
        
        # Save the uploaded image file locally
        file.save(file_path)
        
        # You can add any image processing logic here if needed
        try:
            text_from_image = extract_text_from_image(file_path)
            # Return the results (e.g., image path)
            return jsonify({"filename": file.filename, "path": file_path, "text": text_from_image}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a valid image file."}), 400


# Route to scrape content from URL
@media_processing_bp.route('/scrape_url', methods=['POST'])
def scrape_url():
    # Get URL from request data
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Fetch the webpage content
        response = requests.get(url)
        
        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve the page. Status code: " + str(response.status_code)}), 500
        
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the page (you can modify this based on your needs)
        page_text = soup.get_text()
        
        # Return the extracted content
        return jsonify({"url": url, "extracted_text": page_text.strip()}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 500
    

# @media_processing_bp.route('/process_input', methods=['POST'])
# def process_input():
#     if request.content_type.startswith('multipart/form-data'):
#         # File input
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part in the request"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No file selected"}), 400

#         if file and allowed_file(file.filename):
#             file_path = os.path.join('uploads', file.filename)
#             file.save(file_path)

#              # Check if the file is a .webp image
#             if file.filename.lower().endswith(".webp"):
#                 try:
#                     from PIL import Image
#                     # Convert .webp to .png
#                     with Image.open(file_path) as img:
#                         converted_file_path = os.path.splitext(file_path)[0] + ".png"
#                         img.save(converted_file_path, "PNG")
#                     os.remove(file_path)  # Remove the original .webp file
#                     file_path = converted_file_path  # Update the file path for processing
#                 except Exception as e:
#                     return jsonify({"error": f"Failed to process .webp file: {str(e)}"}), 500
#             return detect_and_process_file(file, file_path)
#         else:
#             return jsonify({"error": "Invalid file type"}), 400

#     elif request.content_type == 'application/json':
#         # JSON input
#         data = request.get_json()
#         return detect_and_process_json(data)

#     else:
#         return jsonify({"error": "Unsupported Content-Type"}), 415


@media_processing_bp.route('/process_input', methods=['POST'])
def process_input():
    if request.content_type.startswith('multipart/form-data'):
        # File input
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            try:
                # Generate a unique filename to avoid overwrites
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads', filename)
                
                # Ensure uploads directory exists
                os.makedirs('uploads', exist_ok=True)
                
                # Save the uploaded file
                file.save(file_path)
                
                # Process the file - no need for separate WebP handling here
                # as it's now handled in extract_text_from_image
                result = detect_and_process_file(file, file_path)
                
                # Clean up the uploaded file after processing
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                return result
                
            except Exception as e:
                # Clean up any files if there's an error
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400

    elif request.content_type == 'application/json':
        # JSON input
        data = request.get_json()
        return detect_and_process_json(data)

    else:
        return jsonify({"error": "Unsupported Content-Type"}), 415