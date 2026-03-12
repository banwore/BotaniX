import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import requests
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AI_MODEL_ID = os.getenv("AI_MODEL_ID", "gemma2:9b")
VISION_MODEL_PATH = os.getenv("VISION_MODEL_PATH", 'model.keras')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini AI Configured!")
else:
    print("⚠️ No Gemini API Key found. Falling back to Ollama.")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


print(f"Loading Vision Model: {VISION_MODEL_PATH}...")
try:
    vision_model = tf.keras.models.load_model(VISION_MODEL_PATH)
    print("✅ Vision Model Loaded!")
except Exception as e:
    print(f"❌ Vision Model Error: {e}")
    try:
        vision_model = tf.keras.models.load_model('best_model_phase1.keras')
        print("✅ Loaded fallback model.")
    except:
        vision_model = None

try:
    with open('disease_data.json', 'r') as f:
        DISEASE_DATA = json.load(f)
    print("✅ Disease database loaded!")
except Exception as e:
    print(f"⚠️ Could not load disease_data.json: {e}")
    DISEASE_DATA = {}


@app.route('/')
def home(): return send_from_directory('.', 'index.html')
@app.route('/<path:path>')
def static_files(path): return send_from_directory('.', path)

@app.route('/wiki-info', methods=['POST'])
def wiki_info():

    data = request.json
    disease = data.get('disease', '')

    if disease in DISEASE_DATA:
        return jsonify(DISEASE_DATA[disease])

    for key in DISEASE_DATA:
        if disease.lower() in key.lower() or key.lower() in disease.lower():
            return jsonify(DISEASE_DATA[key])

    return jsonify({
        'scientific_name': 'Unknown',
        'type': 'Unknown',
        'severity': 'Unknown',
        'overview': 'Information not available for this condition.',
        'symptoms': [],
        'causes': 'Unknown',
        'treatment': {
            'organic': 'Consult a local agricultural extension office.',
            'chemical': 'Consult a licensed pesticide applicator.'
        },
        'prevention': ['Monitor plants regularly', 'Practice good sanitation']
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not vision_model: return jsonify({'error': 'Vision Model not loaded'}), 500
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400

    try:
        file = request.files['image']

        img = Image.open(file).convert('RGB').resize((300, 300))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        print("\n🔮 Vision Model Predicting...")
        predictions = vision_model.predict(img_array, verbose=0)
        score = predictions[0]

        top_indices = np.argsort(score)[-3:][::-1]
        top_results = []
        for idx in top_indices:
            name = CLASS_NAMES[idx].replace('___', ' - ').replace('_', ' ')
            prob = float(score[idx]) * 100
            top_results.append({'class': name, 'probability': prob})

        best_result = top_results[0]
        print(f"🏆 Detected: {best_result['class']}")

        return jsonify({
            'result': best_result['class'],
            'confidence': best_result['probability'],
            'bars_data': top_results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_engine():
    data = request.json
    user_msg = data.get('message', '')
    disease = data.get('disease', 'Unknown')

    prompt = f"""
    SYSTEM ROLE: You are 'BotaniX', an advanced AI Plant Pathologist.
    CONTEXT: Plant diagnosed with: {disease}.
    USER: "{user_msg}"

    INSTRUCTIONS:
    1. Answer simply and scientifically.
    2. Suggest 1 organic treatment if asked for a cure.
    3. Keep it short (max 3 sentences).
    4. Do not use markdown (no **bold**).

    ANSWER:
    """

    # 1. Try Gemini first if Key exists
    if GEMINI_API_KEY:
        try:
            print(f"🧠 Sending to Gemini...")
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            reply_text = response.text
            
            reply_text = reply_text.strip().replace("\n", "<br>")
            return jsonify({'reply': reply_text})
        except Exception as e:
            print(f"❌ Gemini Error: {e}. Trying Ollama fallback...")

    # 2. Fallback to Ollama
    try:
        print(f"💬 Sending to Ollama ({AI_MODEL_ID})...")
        payload = {
            "model": AI_MODEL_ID,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        
        if response.status_code == 200:
            result_json = response.json()
            reply_text = result_json.get('response', '')
            reply_text = reply_text.strip().replace("\n", "<br>")
            return jsonify({'reply': reply_text})
        else:
            return jsonify({'reply': f"⚠️ Error: Local AI responded with status {response.status_code}."})
            
    except Exception as e:
        return jsonify({'reply': "⚠️ Error: Connection to AI engines failed."})

if __name__ == '__main__':
    print("🚀 Starting Local Server...")
    print(f"Make sure you have run 'ollama serve' and 'ollama pull {AI_MODEL_ID}' in a separate terminal!")
    app.run(debug=True, host='0.0.0.0', port=5000)
