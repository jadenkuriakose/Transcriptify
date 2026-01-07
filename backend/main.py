import os
import ssl
import re
import hashlib
import yt_dlp
import torch
import whisper
import groq
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

if os.getenv("DISABLE_SSL_VERIFY") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"

app = Flask(__name__)
CORS(app)

class Processor:
    def __init__(self):
        self.transcriber = None
        self.api_key = os.getenv("API_KEY")
        self.use_groq = bool(self.api_key)

        if self.use_groq:
            self.client = groq.Groq(api_key=self.api_key)
        else:
            self.model = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device=0 if torch.cuda.is_available() else -1
            )

    def get_transcriber(self):
        if self.transcriber is None:
            self.transcriber = whisper.load_model("tiny")
        return self.transcriber

    def chunk(self, text, max_len=1000):
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        curr_chunk = []
        curr_len = 0

        for sentence in sentences:
            length = len(sentence)
            if curr_len + length > max_len and curr_chunk:
                chunks.append(" ".join(curr_chunk))
                curr_chunk = [sentence]
                curr_len = length
            else:
                curr_chunk.append(sentence)
                curr_len += length

        if curr_chunk:
            chunks.append(" ".join(curr_chunk))

        return chunks

    def get_best_chunks(self, chunks, question, max_chunks=3):
        keywords = set(re.findall(r"\w+", question.lower()))
        weights = {
            word: 1.5 if len(word) > 3 else 1.0
            for word in keywords
            if word not in {"what", "when", "where", "who", "how", "why", "does", "did", "is", "are"}
        }

        scored = []
        for chunk in chunks:
            lower = chunk.lower()
            score = sum(weight for word, weight in weights.items() if word in lower)
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:max_chunks]]

    def get_answer(self, chunks, question):
        context = " ".join(chunks)
        prompt = (
            "Based on the following transcript excerpt, answer the question clearly and naturally.\n\n"
            f"Transcript:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

        if self.use_groq:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Answer the question using only facts explicitly stated in the provided transcript. Do not introduce examples, analogies, scenarios, or assumptions that are not directly mentioned in the transcript or that you are sure of"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.6
            )
            return response.choices[0].message.content.strip()

        output = self.model(
            prompt,
            max_new_tokens=300,
            temperature=0.6,
            num_return_sequences=1
        )
        return output[0]["generated_text"].split("Answer:")[-1].strip()

    @lru_cache(maxsize=20)
    def transcribe(self, video_url):
        key = hashlib.md5(video_url.encode()).hexdigest()
        output_path = f"/tmp/{key}.%(ext)s"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }]
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = ydl.prepare_filename(info)
            audio_path = audio_path.replace(".webm", ".mp3").replace(".m4a", ".mp3")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            result = self.get_transcriber().transcribe(
                audio_path,
                fp16=False,
                verbose=False,
                language="en",
                condition_on_previous_text=False,
                task="transcribe"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            os.remove(audio_path)
        except:
            pass

        return result["text"]

    def process_video(self, video_url, question):
        transcript = self.transcribe(video_url)
        return self.process_text(transcript, question)

    def process_text(self, transcript, question):
        chunks = self.chunk(transcript)
        best_chunks = self.get_best_chunks(chunks, question)
        answer = self.get_answer(best_chunks, question)

        return {
            "transcript": transcript,
            "answer": answer
        }

processor = Processor()

@app.route("/process_video", methods=["POST"])
def process_video_route():
    data = request.get_json()
    video_url = data.get("video_url", "").strip()
    question = data.get("question", "").strip()

    if not video_url or not question:
        return jsonify({"error": "Both video_url and question are required"}), 400

    try:
        return jsonify(processor.process_video(video_url, question))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_text", methods=["POST"])
def process_text_route():
    data = request.get_json()
    transcript = data.get("transcript", "").strip()
    question = data.get("question", "").strip()

    if not transcript or not question:
        return jsonify({"error": "Both transcript and question are required"}), 400

    try:
        return jsonify(processor.process_text(transcript, question))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080, debug=True)
