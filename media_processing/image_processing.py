from PIL import Image
import pytesseract
import os
from media_processing.utils import image_ocr
# # Ensure pytesseract can find the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# print(pytesseract.pytesseract.tesseract_cmd)

# def extract_text_from_image(file_path: str) -> dict:
#     """
#     Extract text from an image using Tesseract OCR.

#     Args:
#         file_path (str): Path to the image file.

#     Returns:
#         dict: Dictionary containing extracted text or error message.
#     """
#     try:
#         # Open the image file using Pillow
#         with Image.open(file_path) as img:
#             # Optional: Preprocess image for better OCR (e.g., convert to grayscale)
#             img = img.convert("L")  # Uncomment if needed
            
#             # Extract text using pytesseract
#             extracted_text = pytesseract.image_to_string(img)
#             # extracted_text = image_ocr(img)

#             # Return the result
#             return extracted_text
#     except Exception as e:
#         # Handle errors (e.g., invalid image file)
#         return {"error": f"Failed to process image: {str(e)}"}



from PIL import Image
import pytesseract
import os

def extract_text_from_image(file_path: str) -> dict:
    """
    Extract text from an image using Tesseract OCR, with support for WebP files.
    
    Args:
        file_path (str): Path to the image file.
    
    Returns:
        dict: Dictionary containing extracted text or error message.
    """
    try:
        # Check if the file is WebP
        if file_path.lower().endswith('.webp'):
            # Convert WebP to PNG first
            with Image.open(file_path) as img:
                # Create new filename for PNG
                png_path = os.path.splitext(file_path)[0] + '.png'
                # Convert and save as PNG
                img.convert('RGB').save(png_path, 'PNG')
                # Update file path to use the PNG version
                file_path = png_path
        
        # Process the image (either original non-WebP or converted PNG)
        with Image.open(file_path) as img:
            # Convert to grayscale for better OCR results
            img = img.convert('L')
            
            # Apply some optional preprocessing for better results
            # Increase contrast
            img = img.point(lambda x: 0 if x < 128 else 255, '1')
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(img)
            
            # Clean up the converted PNG if it was created
            if file_path.endswith('.png') and os.path.exists(file_path):
                os.remove(file_path)
            
            # return {
            #     "success": True,
            #     "text": extracted_text.strip(),
            #     "format": "WebP (converted)" if file_path.endswith('.png') else "Standard"
            # }
            return extracted_text.strip()
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to process image: {str(e)}"
        }