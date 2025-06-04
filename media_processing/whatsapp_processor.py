import requests
import os
import tempfile
from urllib.parse import urlparse
import mimetypes
from moviepy.video.io.VideoFileClip import VideoFileClip
from media_processing.video_processing import process_video_file
from media_processing.audio_processing import process_audio_file
from media_processing.image_processing import extract_text_from_image
import re


class WhatsAppMediaProcessor:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        self.supported_image_types = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
        self.supported_video_types = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'}
        self.supported_audio_types = {'mp3', 'wav', 'aac', 'm4a', 'ogg'}
        
        # Ensure upload folder exists
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def detect_media_type(self, url, headers_dict=None):
        """
        Detect if URL contains image, video, or audio based on URL patterns and content type
        """
        try:
            url_lower = url.lower()
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            # Check file extension in URL first
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv']
            audio_extensions = ['.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac', '.wma']
            
            if any(ext in path for ext in image_extensions):
                return 'image'
            elif any(ext in path for ext in video_extensions):
                return 'video'
            elif any(ext in path for ext in audio_extensions):
                return 'audio'
            
            # Try HEAD request to get content type with authorization
            try:
                head_headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                if headers_dict:
                    head_headers.update(headers_dict)
                
                head_response = requests.head(url, headers=head_headers, timeout=10, allow_redirects=True)
                content_type = head_response.headers.get('content-type', '').lower()
                
                if content_type.startswith('image/'):
                    return 'image'
                elif content_type.startswith('video/'):
                    return 'video'
                elif content_type.startswith('audio/'):
                    return 'audio'
                
            except Exception as e:
                print(f"Warning: Could not determine content type via HEAD request: {e}")
            
            # Default fallback based on URL structure
            if any(keyword in url_lower for keyword in ['image', 'img', 'photo', 'pic']):
                return 'image'
            elif any(keyword in url_lower for keyword in ['video', 'vid', 'movie']):
                return 'video'
            elif any(keyword in url_lower for keyword in ['audio', 'sound', 'music']):
                return 'audio'
            
            # Final fallback - assume image for WhatsApp media
            return 'image'
            
        except Exception as e:
            print(f"Error detecting media type: {e}")
            return 'image'  # Default fallback
    
    def download_whatsapp_media(self, url, access_token, media_type):
        """
        Download media file from WhatsApp URL with authorization token
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Authorization': f'Bearer {access_token}',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
            response.raise_for_status()
            
            # Determine file extension from content type or URL
            content_type = response.headers.get('content-type', '').lower()
            
            if media_type == 'image':
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                elif 'webp' in content_type:
                    ext = '.webp'
                else:
                    # Try to infer from URL or default
                    url_lower = url.lower()
                    if 'png' in url_lower:
                        ext = '.png'
                    elif 'gif' in url_lower:
                        ext = '.gif'
                    elif 'webp' in url_lower:
                        ext = '.webp'
                    else:
                        ext = '.jpg'  # Default
                        
            elif media_type == 'video':
                if 'mp4' in content_type:
                    ext = '.mp4'
                elif 'webm' in content_type:
                    ext = '.webm'
                elif 'mov' in content_type or 'quicktime' in content_type:
                    ext = '.mov'
                else:
                    ext = '.mp4'  # Default
                    
            else:  # audio
                if 'mp3' in content_type or 'mpeg' in content_type:
                    ext = '.mp3'
                elif 'wav' in content_type:
                    ext = '.wav'
                elif 'aac' in content_type:
                    ext = '.aac'
                elif 'm4a' in content_type:
                    ext = '.m4a'
                elif 'ogg' in content_type:
                    ext = '.ogg'
                else:
                    ext = '.mp3'  # Default
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=self.upload_folder)
            
            # Download with progress tracking for large files
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Optional: Print progress for large files
                    if total_size > 0 and total_size > 1024 * 1024:  # > 1MB
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rDownloading: {progress:.1f}%", end="", flush=True)
            
            if total_size > 1024 * 1024:
                print()  # New line after progress
            
            temp_file.close()
            
            # Verify file was downloaded successfully
            if os.path.getsize(temp_file.name) == 0:
                os.unlink(temp_file.name)
                return None, "Downloaded file is empty"
            
            return temp_file.name, None
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return None, "Unauthorized - Invalid or expired access token"
            elif e.response.status_code == 403:
                return None, "Forbidden - Access denied to this media"
            elif e.response.status_code == 404:
                return None, "Media not found or URL expired"
            else:
                return None, f"HTTP error {e.response.status_code}: {str(e)}"
        except requests.exceptions.Timeout:
            return None, "Download timeout - file too large or connection too slow"
        except requests.exceptions.RequestException as e:
            return None, f"Network error downloading media: {str(e)}"
        except Exception as e:
            return None, f"Error downloading media: {str(e)}"
    
    def process_media_file(self, file_path, media_type):
        """
        Process the media file based on its type with better error handling
        """
        try:
            if media_type == 'image':
                result = extract_text_from_image(file_path)
                # Get image info
                file_size = os.path.getsize(file_path)
                
                return {
                    'type': 'image',
                    'text': result,
                    'file_size': file_size,
                    'summary': f"Image ({file_size/1024:.1f}KB) - Extracted text: {result[:200]}..." if len(result) > 200 else f"Image ({file_size/1024:.1f}KB) - Extracted text: {result}"
                }
            
            elif media_type == 'video':
                # Get video info
                video_clip = VideoFileClip(file_path)
                duration = video_clip.duration
                fps = video_clip.fps
                resolution = video_clip.size
                video_clip.close()
                
                # Process video for transcript
                transcript_result = process_video_file(file_path)
                transcript = transcript_result.text if hasattr(transcript_result, 'text') else str(transcript_result)
                
                file_size = os.path.getsize(file_path)
                
                return {
                    'type': 'video',
                    'duration': duration,
                    'fps': fps,
                    'resolution': resolution,
                    'file_size': file_size,
                    'transcript': transcript,
                    'summary': f"Video ({file_size/1024/1024:.1f}MB, {duration:.1f}s, {resolution[0]}x{resolution[1]}) - Transcript: {transcript[:300]}..." if len(transcript) > 300 else f"Video ({file_size/1024/1024:.1f}MB, {duration:.1f}s, {resolution[0]}x{resolution[1]}) - Transcript: {transcript}"
                }
            
            elif media_type == 'audio':
                transcript = process_audio_file(file_path)
                file_size = os.path.getsize(file_path)
                
                # Try to get audio duration
                try:
                    from moviepy.audio.io.AudioFileClip import AudioFileClip
                    audio_clip = AudioFileClip(file_path)
                    duration = audio_clip.duration
                    audio_clip.close()
                except:
                    duration = None
                
                return {
                    'type': 'audio',
                    'duration': duration,
                    'file_size': file_size,
                    'transcript': transcript,
                    'summary': f"Audio ({file_size/1024:.1f}KB{f', {duration:.1f}s' if duration else ''}) - Transcript: {transcript[:300]}..." if len(transcript) > 300 else f"Audio ({file_size/1024:.1f}KB{f', {duration:.1f}s' if duration else ''}) - Transcript: {transcript}"
                }
            
            else:
                return {'error': f'Unsupported media type: {media_type}'}
                
        except Exception as e:
            return {'error': f'Error processing {media_type}: {str(e)}'}
    
    def process_whatsapp_media_url(self, media_url, access_token):
        """
        Main method to process WhatsApp media URL and extract/summarize content
        """
        try:
            print(f"Processing WhatsApp media: {media_url}")
            
            # Detect media type (pass headers for authorization if needed)
            auth_headers = {'Authorization': f'Bearer {access_token}'}
            media_type = self.detect_media_type(media_url, auth_headers)
            print(f"Detected media type: {media_type}")
            
            # Download media with authorization
            file_path, download_error = self.download_whatsapp_media(media_url, access_token, media_type)
            if download_error:
                print(f"Download error: {download_error}")
                return {'error': download_error}
            
            print(f"Downloaded to: {file_path}")
            
            # Process media
            processing_result = self.process_media_file(file_path, media_type)
            processing_result['url'] = media_url
            processing_result['file_path'] = file_path
            
            print(f"Processing result: {processing_result.get('summary', 'No summary')}")
            
            # Clean up temporary file
            try:
                os.unlink(file_path)
                print(f"Cleaned up temporary file: {file_path}")
            except Exception as cleanup_error:
                print(f"Warning: Could not clean up file {file_path}: {cleanup_error}")
            
            return {
                'input_url': media_url,
                'url_type': 'whatsapp_media',
                'media_type': media_type,
                'result': processing_result,
                'success': 'error' not in processing_result
            }
            
        except Exception as e:
            return {'error': f'Error processing WhatsApp media URL: {str(e)}'}


# # Example usage:
# if __name__ == "__main__":
#     processor = WhatsAppMediaProcessor()
    
#     # Test with your WhatsApp media URL
#     test_url = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=1524586061836361&ext=1749030618&hash=ATvvK3QmPNeu8gROrku-5epiIzbbW5vQmRQPyu77DTKbeA"
#     test_token = "EAAZAfjDLptMYBO5TKFOyTWbyQ1GD1ZC4U9MKy6vXhv32MOLjXtOz5MwusFZAz4WBRkwe0aw3JzsHE3yT8eoQGZCKZBVyexFe6rK6j2LaNgtQ8Ruva0ZAgKwOKii1YZB7rPEcKgjugcLcxkn5l5GVL7fHvYrVhRjwxkZCBhZCbDIN9x2HIe5DINUvGHqBwebb9sjzT9wZDZD"
    
#     result = processor.process_whatsapp_media_url(test_url, test_token)
    
#     print("Result:")
#     print(result)