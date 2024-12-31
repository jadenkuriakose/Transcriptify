import ssl
import os
import re
import yt_dlp
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import torch
from transformers import pipeline
import groq
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

app = Flask(__name__)
CORS(app)

class Processor:
    def __init__(self):
        self.transcriber = whisper.load_model("tiny")
        self.use_groq = os.getenv("API_KEY") is not None
        
        if self.use_groq:
            self.client = groq.Groq(api_key=os.getenv("API_KEY"))
        else:
            self.model = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device=0 if torch.cuda.is_available() else -1
            )
            
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )

    def chunk(self, text, max_len=1000):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        curr_chunk = []
        curr_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if curr_len + sentence_len > max_len and curr_chunk:
                chunks.append(' '.join(curr_chunk))
                curr_chunk = [sentence]
                curr_len = sentence_len
            else:
                curr_chunk.append(sentence)
                curr_len += sentence_len
        
        if curr_chunk:
            chunks.append(' '.join(curr_chunk))
        return chunks

    def get_best_chunks(self, chunks, question, max_chunks=3):
        keywords = set(re.findall(r'\w+', question.lower()))
        weights = {word: 1.5 if len(word) > 3 else 1.0 
                 for word in keywords 
                 if word not in {'what', 'when', 'where', 'who', 'how', 'why', 'does', 'did', 'is', 'are'}}
        
        scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(weight for word, weight in weights.items() 
                       if word in chunk_lower)
            scores.append((chunk, score))
        
        return [chunk for chunk, score in sorted(scores, key=lambda x: x[1], reverse=True)[:max_chunks]]

    def get_answer(self, chunks, question):
        text = ' '.join(chunks)
        prompt = f"""Based on the following video transcript excerpt, provide a detailed, natural answer to the question. 
        Use complete sentences and focus on the most relevant information.
        
        Transcript: {text}
        
        Question: {question}
        
        Answer: """
        
        if self.use_groq:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Use whatever text or transcript is provided to answer questions in a human way only using context from the media source and slight outside knowledge to output a human response that may have some minor unnoticeable grammatical issues."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        else:
            response = self.model(prompt, max_new_tokens=500, temperature=0.7, num_return_sequences=1)
            answer = response[0]['generated_text'].split("Answer: ")[-1].strip()
        
        return answer.strip()

    @lru_cache(maxsize=100)
    def transcribe(self, video_url):
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': '/tmp/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
        
        result = self.transcriber.transcribe(
            audio,
            verbose=False,
            fp16=False,
            language='en',
            condition_on_previous_text=False,
            task="transcribe"
        )
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            os.unlink(audio)
        except:
            pass
            
        return result['text']

    def process_video(self, video_url, question):
        transcript = self.transcribe(video_url)
        return self.process_text(transcript, question)

    def process_text(self, transcript, question):
        chunks = self.chunk(transcript)
        best_chunks = self.get_best_chunks(chunks, question)
        answer = self.get_answer(best_chunks, question)
        
        with torch.no_grad():
            sentiment = self.sentiment(answer[:500])[0]
        
        return {
            "transcript": transcript,
            "answer": answer,
            "sentiment": sentiment
        }

processor = Processor()

@app.route("/process_video", methods=["POST"])
def process_video_route():
    data = request.get_json()
    video_url = data.get("video_url", "").strip()
    question = data.get("question", "").strip()
    
    if not video_url or not question:
        return jsonify({"error": "Both video URL and question are required."}), 400
    
    try:
        result = processor.process_video(video_url, question)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/process_text", methods=["POST"])
def process_text_route():
    data = request.get_json()
    transcript = data.get("transcript", "").strip()
    question = data.get("question", "").strip()
    
    try:
        result = processor.process_text(transcript, question)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)
