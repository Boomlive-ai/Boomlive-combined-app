from pydub import AudioSegment
import speech_recognition as sr
import os

def process_audio_file(file_path: str) -> str:
    recognizer = sr.Recognizer()

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file does not exist at path: {file_path}")

        # Convert to WAV if necessary
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Ensure mono audio with 16 kHz
        wav_path = file_path.replace(os.path.splitext(file_path)[1], ".wav")
        audio.export(wav_path, format="wav")

        # Perform speech recognition
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        # Clean up converted WAV file
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return transcription
    except Exception as e:
        return f"Failed to process audio: {e}"
