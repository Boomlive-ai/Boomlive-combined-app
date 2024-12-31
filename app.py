from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Import blueprints from the modules
from factcheck.routes import factcheck_bp
from chatbot.routes import chatbot_bp
from media_processing.routes import media_processing_bp

# Initialize the app
app = Flask(__name__)
CORS(app)
load_dotenv()

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register blueprints
app.register_blueprint(factcheck_bp, url_prefix='/factcheck')
app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
app.register_blueprint(media_processing_bp, url_prefix='/media')

@app.route('/')
def index():
    return {"message": "Welcome to the Boom app!"}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
