from IPython.display import Markdown
import textwrap
import os
import google.generativeai as genai
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
llm  = genai.GenerativeModel('gemini-1.5-pro-latest')

def image_ocr(img):
    response = llm.generate_content([
        "what is the claim of text in this image?", 
        img
    ])

    if response.candidates:
        # Extract the text from the first candidate
        raw_text = response.candidates[0].content.parts[0].text
        
        return raw_text
       
    else:
        return {"error": "No text detected."}
  
  
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))