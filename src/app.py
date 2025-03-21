import sys
import os
import io
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from flask_cors import CORS  # ✅ Import CORS to fix resource blocking
from gtts import gTTS
from views import views  # Importing views.py
from response_generator import generate_response  # ✅ Import response generator
from Intent_classification import intent_prediction  

# Initialize Flask app
app = Flask(
    __name__,
    static_folder='../Frontend/static',  # Path to static files (Live2D model, JS, CSS)
    template_folder='../Frontend/templates'  # Path to templates (HTML)
)

CORS(app)  # ✅ Enable CORS for all routes

# Register the views blueprint
app.register_blueprint(views)

# 🟢 Serve the chatbot frontend
@app.route('/')
def index():
    return render_template('index.html')

# 🟢 API route for chatbot communication
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({"error": "Empty message received."}), 400
    
    user_id = 123  # Dummy ID

    # ✅ Debugging: Print detected intent
    intent = intent_prediction(user_message)
    print(f"🔍 User Message: {user_message}")
    print(f"🎯 Predicted Intent: {intent}")  # ✅ Check if intent is detected correctly

    # ✅ Debug: If intent is "DEFAULT", print a warning
    if intent == "DEFAULT":
        print("⚠ WARNING: Intent classification failed! Fallback triggered.")

    # ✅ Fix: Call `generate_response` correctly
    bot_response = generate_response(user_id, user_message)

    # ✅ Debug Response Generation
    print(f"🤖 Bot Response: {bot_response}")

    # ✅ Convert response to speech and return audio
    audio_stream = text_to_speech(bot_response)

    return send_file(audio_stream, mimetype="audio/mp3")

# ✅ Function to convert text to speech and return audio in memory
def text_to_speech(response_text):
    if not response_text:
        response_text = "I'm sorry, I could not understand the question."
    
    tts = gTTS(text=response_text, lang='en')

    # Store the speech output in memory instead of a file
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)  # Rewind the buffer to the beginning
    return audio_stream

# 🟢 Serve static files (Fix Live2D Model Loading Issue)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '../Frontend/static'), filename)

if __name__ == '__main__':
    app.run(debug=True)
