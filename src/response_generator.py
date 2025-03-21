import sys
import os
import random
import json
import speech_recognition as sr
from gtts import gTTS
import io
import pygame

# Setting up the directory to the main folder "College_AI"
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(src_path)

from DataBase.context_manager import store_context_db, get_context_db  # Import SQLite context functions
from src.Intent_classification import intent_prediction  # Import intent prediction function
from src.Entity_extraction import entity_extraction  # Import entity extraction function

# Load responses from JSON file
def load_responses(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

file_path = os.path.join(os.path.dirname(__file__), "../response_data/responses.json")
response_data = load_responses(file_path)

# ✅ Generate response based on query or follow-up (No follow-up storage here)
def generate_response(user_id, user_query):

    #******************* This Block Is performing NLP Tasks *******************#
    intent = intent_prediction(user_query)  # Predict intent

    try:
        entities = entity_extraction(user_query)  # Extract entities
    except Exception as e:
        print(f"Error in entity extraction: {e}")
        entities = []  # Default to empty entity list
    #***************************************************************************#

    #******************* This Block Is Fetching Response *******************#
    responses = []

    if intent in response_data:
        intent_responses = response_data[intent]

        if not entities:
            response = random.choice(intent_responses.get("default", ["I'm not sure."]))
        else:
            response = []
            for entity in entities:
                if entity in intent_responses:
                    response.append(random.choice(intent_responses[entity]))
                else:
                    response.append(random.choice(intent_responses.get("default", ["I'm not sure."])))
            response = " ".join(response)

        responses.append(response)

        # Store intent and entities in database disregarding the existance followup question
        handle_context(user_id, "store", intent, entities, user_query)  # Store intent, entites and User_query

    else:
        responses.append(random.choice(response_data["DEFAULT"]["default"]))

    return "\n".join(responses)
    #***************************************************************************#
    
# ✅ Find follow-up question in JSON (Does NOT store anything)
def followUp_question(intent):
    """
    Retrieves follow-up question from JSON if available.
    """
    intent_responses = response_data.get(intent, {})
    followup_question = intent_responses.get("followup")
    return random.choice(followup_question) if followup_question else None

# ✅ Store & retrieve only intent & entities (No follow-up question storage)
def handle_context(user_id, action, intent=None, entities=None,last_response=None):
    """
    Manages user-specific context for conversation continuity.
    Stores only user_id, intent, and entities.
    """
    try:
        if action == "store":
            store_context_db(user_id, intent, entities, last_response)  # Store intent & entities only
        elif action == "retrieve":
            return get_context_db(user_id)  # Retrieve intent & entities only
        elif action == "clear":
            store_context_db(user_id, "", {})  # Clear intent & entities
    except Exception as e:
        print(f"Error in context handling: {e}")

# Function to convert text to speech and play it from memory without saving a file
def text_to_speech(response_text):
    """Convert text to speech and play it from memory without saving a file."""
    if not response_text:
        response_text = "I'm sorry, I could not understand the question."
    tts = gTTS(text=response_text, lang='en', slow=False)
    
    # Store the speech output in memory instead of a file
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    print("Speaking...")
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load audio from memory
    audio_stream.seek(0)
    audio_data = audio_stream.read()
    audio_stream.seek(0)
    
    # Play the audio using pygame
    pygame.mixer.music.load(io.BytesIO(audio_data), "mp3")
    pygame.mixer.music.play()
    
    # Wait until playback is done
    while pygame.mixer.music.get_busy():
        continue

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("I'm Listening...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        print("Could not understand the audio, please try again.")
        return None
    except sr.RequestError:
        print("Could not request results, check your internet connection.")
        return None

# ✅ Function to handle follow-up questions
def handle_followup(user_id, context):
    stored_intent = context.get("intent")
    if stored_intent:
        followup_questions = followUp_question(stored_intent)  # Get follow-up question from JSON
        if followup_questions:
            # Asistant will ask the folloup Question
            print(f"{followup_questions}")
            text_to_speech(followup_questions)

            user_query = speech_to_text()  # Take user's input for the follow-up
            print(f"User ({user_id}): {user_query}")
            
            if user_query is None:
               text_to_speech("Please try again.")
               return None, True
            
            # ✅ If user says "yes", continue follow-up process
            if user_query.lower() in ["yes", "yeah", "yep", "sure", "okay", "yes please"]:
                print("Entering yes statement in handle_followup function... line 157")
                user_query = followup_questions
                print(f"User ({user_id}): {user_query}")
                response = generate_response(user_id, user_query)   # new context is stored here again.
                print("generating response using followup question in handle_followup function... line 160")
                print(f"Assistant ({user_id}): {response}")
                text_to_speech(response)
                return response, True
                # next_followup = followUp_question(stored_intent)  # Get next follow-up

            # ✅ If user says "no", clear context & move on
            elif user_query.lower() in ["no", "nope", "nah", "not really"]:
                handle_context(user_id, "clear")
                text_to_speech("Alright, let me know if you have any other questions.")
                return "Alright, let me know if you have any other questions.", False

            else:
                response = generate_response(user_id, user_query)   # the context is stored here again.
                print(f"Assistant ({user_id}): {response}")
                text_to_speech(response)
                return response, True  # End follow-up
    return None, False

# def followup_question(intent):
#     followup_question = followup_questions.get(intent)

# ✅ Main chatbot function with improved conversation flow
def chatbot_flow():
    while True:
        user_id = 123  # Dummy user ID for context management

        while True:
            print(f"You ({user_id}): Say something... (Say 'quit', 'bye', or 'exit' to quit)")
            user_query = speech_to_text()
            if user_query is None:
                text_to_speech("Please try again.")
                continue
            elif user_query.lower() in ["quit", "bye", "exit"]:
                print(f"Assistant ({user_id}): Goodbye!")
                text_to_speech("Goodbye!, Have a nice day.")
                handle_context(user_id, "clear")
                break
            else:
                # ✅ Step 2: Process the user query normally
                response = generate_response(user_id, user_query)
                print(f"Assistant ({user_id}): {response}")
                text_to_speech(response)
            
            # ✅ Step 1: Check if a follow-up exists
            context = handle_context(user_id, "retrieve")
            if context:
                print("Entering handling followup function... line 221")
                _ , ask_followup = handle_followup(user_id, context)
                if ask_followup:
                    print("Asking follow-up question... line 224")
                    context = handle_context(user_id, "retrieve")
                    handle_followup(user_id, context)
                    continue
                

# ✅ Run the chatbot
if __name__ == "__main__":
    chatbot_flow()