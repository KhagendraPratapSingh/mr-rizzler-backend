import os
import base64
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

TONE_PROMPTS = {
    "flirty": "playful, confident, and a little flirty",
    "funny": "witty and humorous, aimed at making them laugh",
    "smooth": "smooth, charming, and effortlessly confident",
    "savage": "bold, teasing, and cheeky -- but never mean or disrespectful",
    "respectful": "warm, genuine, and respectful",
}

SYSTEM_INSTRUCTIONS = """You are a dating conversation assistant. You will be shown a screenshot \
of a chat conversation. Your job is to:
1. Read and understand the conversation context from the image.
2. Identify what the last message said.
3. Generate 4 distinct reply suggestions the user could send next, in the requested tone.

Rules:
- Replies must always be safe, respectful, and appropriate.
- Keep each reply under 25 words.
- Make replies sound natural and human, not robotic or cheesy.
- Return ONLY valid JSON in this exact format, nothing else before or after:
{
  "context_summary": "one sentence summary of what is being discussed",
  "suggestions": ["reply 1", "reply 2", "reply 3", "reply 4"]
}"""


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if not GEMINI_API_KEY:
        return jsonify({"error": "Server missing GEMINI_API_KEY"}), 500

    tone = request.form.get("tone", "smooth")
    tone_description = TONE_PROMPTS.get(tone, TONE_PROMPTS["smooth"])

    image_file = request.files["image"]
    image_bytes = image_file.read()
    mime_type = image_file.mimetype
    if not mime_type or mime_type == "application/octet-stream":
        mime_type = "image/jpeg"


    try:
        image_part = {
            "mime_type": mime_type,
            "data": image_bytes
        }

        prompt = f"{SYSTEM_INSTRUCTIONS}\n\nGenerate replies in a {tone_description} tone."

        response = model.generate_content(
            [prompt, image_part],
            generation_config=genai.types.GenerationConfig(
                temperature=0.9,
                response_mime_type="application/json",
            )
        )

        text_output = response.text
        parsed = json.loads(text_output)
        return jsonify(parsed)

    except Exception as e:
        error_message = str(e)

        print(f"ACTUAL ERROR: {error_message}", flush=True)

        if "quota" in error_message.lower():
            return jsonify({
                "error": "Daily AI limit reached. Please try again tomorrow."
            }), 429

        return jsonify({
            "error": "AI service temporarily unavailable."
        }), 502

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)