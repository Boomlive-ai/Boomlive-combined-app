from PIL import Image
# import pytesseract
import os
from media_processing.utils import image_ocr
# # Ensure pytesseract can find the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# print(pytesseract.pytesseract.tesseract_cmd)

def extract_text_from_image(file_path: str) -> dict:
    """
    Extract text from an image using Tesseract OCR.

    Args:
        file_path (str): Path to the image file.

    Returns:
        dict: Dictionary containing extracted text or error message.
    """
    try:
        # Open the image file using Pillow
        with Image.open(file_path) as img:
            # Optional: Preprocess image for better OCR (e.g., convert to grayscale)
            # img = img.convert("L")  # Uncomment if needed
            
            # Extract text using pytesseract
            # extracted_text = pytesseract.image_to_string(img)
            extracted_text = image_ocr(img)

            # Return the result
            return {"text": extracted_text}
    except Exception as e:
        # Handle errors (e.g., invalid image file)
        return {"error": f"Failed to process image: {str(e)}"}
