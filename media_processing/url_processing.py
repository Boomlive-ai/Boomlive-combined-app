import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

# Web scraping function to extract text from a URL
def scrape_url(url):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        
        if response.status_code != 200:
            return jsonify({"error": f"Failed to retrieve the page. Status code: {response.status_code}"}), 500
        
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from the page (you can modify this based on your needs)
        page_text = soup.get_text()
        
        # Return the extracted content
        return jsonify({"url": url, "extracted_text": page_text.strip()}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to scrape URL: {str(e)}"}), 500