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
from bs4 import BeautifulSoup


class TwitterMediaProcessor:
    def __init__(self, upload_folder='uploads'):
        self.upload_folder = upload_folder
        self.supported_image_types = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
        self.supported_video_types = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'}
        self.supported_audio_types = {'mp3', 'wav', 'aac', 'm4a', 'ogg'}
        
        # Ensure upload folder exists
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def is_direct_media_url(self, url):
        """
        Check if the URL is a direct media URL (not a Twitter post URL)
        """
        media_domains = ['pbs.twimg.com', 'video.twimg.com', 'ton.twimg.com']
        parsed_url = urlparse(url)
        
        # Check if it's a direct media URL
        if any(domain in parsed_url.netloc for domain in media_domains):
            return True
        
        # Check if URL has media file extension
        path = parsed_url.path.lower()
        media_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.mov', '.avi', '.mkv', '.webm', '.mp3', '.wav', '.aac', '.m4a', '.ogg']
        if any(ext in path for ext in media_extensions):
            return True
            
        return False
    
    def extract_media_urls_from_twitter(self, twitter_url):
        """
        Extract media URLs from Twitter/X post
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = requests.get(twitter_url, headers=headers, timeout=30)
            if response.status_code != 200:
                return None, f"Failed to fetch Twitter page: {response.status_code}"
            
            soup = BeautifulSoup(response.content, 'html.parser')
            media_urls = set()  # Use set to avoid duplicates
            
            # Look for various media URL patterns in Twitter/X
            # Images - look for img tags with Twitter media domains
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src', '')
                if src and any(domain in src for domain in ['pbs.twimg.com', 'video.twimg.com']):
                    if not src.startswith('http'):
                        src = 'https:' + src
                    # Remove size parameters to get original quality
                    src = re.sub(r'[?&](name|format)=[^&]*', '', src)
                    media_urls.add(src)
            
            # Videos - look for video tags
            video_tags = soup.find_all('video')
            for video in video_tags:
                src = video.get('src', '')
                poster = video.get('poster', '')
                
                if src:
                    if not src.startswith('http'):
                        src = 'https:' + src
                    media_urls.add(src)
                
                if poster and any(domain in poster for domain in ['pbs.twimg.com', 'video.twimg.com']):
                    if not poster.startswith('http'):
                        poster = 'https:' + poster
                    media_urls.add(poster)
            
            # Look for meta tags with media content
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                property_name = meta.get('property', '') or meta.get('name', '')
                content = meta.get('content', '')
                
                if content and property_name in ['og:image', 'og:video', 'twitter:image', 'twitter:player:stream']:
                    if any(domain in content for domain in ['pbs.twimg.com', 'video.twimg.com', 'ton.twimg.com']):
                        if not content.startswith('http'):
                            content = 'https:' + content
                        media_urls.add(content)
            
            # Look for JSON-LD structured data
            json_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        # Look for image or video URLs in structured data
                        for key, value in data.items():
                            if key in ['image', 'video', 'contentUrl', 'thumbnailUrl']:
                                if isinstance(value, str) and any(domain in value for domain in ['pbs.twimg.com', 'video.twimg.com']):
                                    media_urls.add(value)
                except:
                    continue
            
            return list(media_urls), None
            
        except Exception as e:
            return None, f"Error extracting media URLs: {str(e)}"
    
    def detect_media_type(self, url):
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
            
            # Twitter-specific URL patterns
            if 'pbs.twimg.com' in url_lower:
                # Most pbs.twimg.com URLs are images unless explicitly video
                if 'video' in url_lower or 'tweet_video' in url_lower:
                    return 'video'
                else:
                    return 'image'
            elif 'video.twimg.com' in url_lower:
                return 'video'
            elif 'ton.twimg.com' in url_lower:
                return 'audio'  # Twitter audio spaces
            
            # If no clear pattern, try HEAD request to get content type
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': 'https://twitter.com/'
                }
                
                head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
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
            
            # Final fallback - assume image for Twitter media
            return 'image'
            
        except Exception as e:
            print(f"Error detecting media type: {e}")
            return 'image'  # Default fallback
    
    def download_media(self, url, media_type):
        """
        Download media file from URL with better error handling and format detection
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://twitter.com/',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            # For Twitter media, sometimes we need to modify the URL for better quality
            if 'pbs.twimg.com' in url and media_type == 'image':
                # Remove size restrictions and get original quality
                url = re.sub(r'[?&](name|format)=[^&]*', '', url)
                if '?' not in url:
                    url += '?format=jpg&name=orig'
                else:
                    url += '&format=jpg&name=orig'
            
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
                    # Try to infer from URL
                    url_lower = url.lower()
                    if '.png' in url_lower:
                        ext = '.png'
                    elif '.gif' in url_lower:
                        ext = '.gif'
                    elif '.webp' in url_lower:
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
    
    def process_twitter_url(self, twitter_url):
        """
        Main method to process Twitter URL and extract/summarize all media
        Handles both Twitter post URLs and direct media URLs
        """
        try:
            print(f"Processing: {twitter_url}")
            
            # Check if it's a direct media URL
            if self.is_direct_media_url(twitter_url):
                print("Detected direct media URL")
                media_urls = [twitter_url]
            else:
                print("Detected Twitter post URL, extracting media...")
                # Extract media URLs from Twitter post
                media_urls, error = self.extract_media_urls_from_twitter(twitter_url)
                if error:
                    return {'error': error}
            
            print(f"Found {len(media_urls) if media_urls else 0} media URLs: {media_urls}")
            
            if not media_urls:
                return {'error': 'No media found in the Twitter post'}
            
            results = []
            
            for i, url in enumerate(media_urls, 1):
                print(f"Processing media {i}/{len(media_urls)}: {url}")
                
                # Detect media type
                media_type = self.detect_media_type(url)
                print(f"Detected media type: {media_type}")
                
                # Download media
                file_path, download_error = self.download_media(url, media_type)
                if download_error:
                    print(f"Download error: {download_error}")
                    results.append({
                        'url': url,
                        'error': download_error
                    })
                    continue
                
                print(f"Downloaded to: {file_path}")
                
                # Process media
                processing_result = self.process_media_file(file_path, media_type)
                processing_result['url'] = url
                processing_result['file_path'] = file_path
                
                results.append(processing_result)
                print(f"Processing result: {processing_result.get('summary', 'No summary')}")
                
                # Clean up temporary file
                try:
                    os.unlink(file_path)
                    print(f"Cleaned up temporary file: {file_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up file {file_path}: {cleanup_error}")
            
            return {
                'input_url': twitter_url,
                'url_type': 'direct_media' if self.is_direct_media_url(twitter_url) else 'twitter_post',
                'media_count': len(results),
                'media_results': results,
                'success_count': len([r for r in results if 'error' not in r]),
                'error_count': len([r for r in results if 'error' in r])
            }
            
        except Exception as e:
            return {'error': f'Error processing URL: {str(e)}'}


# # Example usage:
# if __name__ == "__main__":
#     processor = TwitterMediaProcessor()
    
#     # Test with your direct media URL
#     test_url = "https://pbs.twimg.com/media/Gsbe8v9WwAATjN6.jpg"
#     result = processor.process_twitter_url(test_url)
    
#     print("Result:")
#     print(result)