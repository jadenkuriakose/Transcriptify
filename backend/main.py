import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

def analyze_input_length(prompt):
    input_length = len(prompt.split())
    if input_length < 20:
        return 50
    elif input_length < 50:
        return 150
    else:
        return 300

def generate_human_answer(prompt):
    api_key = os.getenv("API_KEY")
    if not api_key:
        return "API key not found.", 500

    client = Groq(api_key=api_key)

    try:
        messages = [{"role": "user", "content": prompt}]
        max_tokens = analyze_input_length(prompt)
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9,
            stream=True,
            stop=None,
        )

        bot_response = "".join(
            chunk.choices[0].delta.content or "" for chunk in completion
        ).strip()

        bot_response = re.sub(r"\bAs an AI\b", "", bot_response)
        bot_response = re.sub(r"(?i)\bAI\b", "", bot_response)
        bot_response = bot_response.strip()

        if not bot_response:
            return {
                "response": "Answer is not in the text, but a possible answer might be...",
            }, 200
        
        return {
            "response": bot_response,
        }, 200

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route("/answer_query", methods=["POST"])
def answer_query():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    response, status_code = generate_human_answer(prompt)
    if status_code == 200:
        return jsonify(response), 200
    else:
        return jsonify({"error": response}), status_code

if __name__ == "__main__":
    app.run(debug=True, port=8080)
