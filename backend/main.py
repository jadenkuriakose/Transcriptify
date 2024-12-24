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

class VideoProcessor:
    def __init__(self):
        self.whisper_model = whisper.load_model("tiny")
        self.use_groq = os.getenv("API_KEY") is not None
        
        if self.use_groq:
            self.groq_client = groq.Groq(api_key=os.getenv("API_KEY"))
        else:
            self.local_model = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device=0 if torch.cuda.is_available() else -1
            )
            
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )

    def chunk_text(self, text, max_chunk_length=1000):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def get_relevant_chunks(self, chunks, question, max_chunks=3):
        question_keywords = set(re.findall(r'\w+', question.lower()))
        keyword_weights = {word: 1.5 if len(word) > 3 else 1.0 
                         for word in question_keywords 
                         if word not in {'what', 'when', 'where', 'who', 'how', 'why', 'does', 'did', 'is', 'are'}}
        
        chunk_scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(weight for word, weight in keyword_weights.items() 
                       if word in chunk_lower)
            chunk_scores.append((chunk, score))
        
        return [chunk for chunk, score in sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:max_chunks]]

    def generate_response(self, transcript_chunks, question):
        relevant_text = ' '.join(transcript_chunks)
        prompt = f"""Based on the following video transcript excerpt, provide a detailed, natural answer to the question. 
        Use complete sentences and focus on the most relevant information.
        
        Transcript: {relevant_text}
        
        Question: {question}
        
        Answer: """
        
        if self.use_groq:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides human answers that sound conversational and have a minor grammatical issue based off text and video transcripts."},
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                max_tokens=500,
                temperature=0.7
            )
            answer = response.choices[0].message.content
        else:
            response = self.local_model(prompt, max_new_tokens=500, temperature=0.7, num_return_sequences=1)
            answer = response[0]['generated_text'].split("Answer: ")[-1].strip()
        
        return answer.strip()

    @lru_cache(maxsize=100)
    def download_and_transcribe(self, video_url):
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
            info_dict = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
        
        result = self.whisper_model.transcribe(
            audio_file,
            verbose=False,
            fp16=False,
            language='en',
            condition_on_previous_text=False,
            task="transcribe"
        )
        
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            os.unlink(audio_file)
        except:
            pass
            
        return result['text']

    def process_video(self, video_url, question):
        transcript = self.download_and_transcribe(video_url)
        return self.process_text(transcript, question)

    def process_text(self, transcript, question):
        chunks = self.chunk_text(transcript)
        relevant_chunks = self.get_relevant_chunks(chunks, question)
        answer = self.generate_response(relevant_chunks, question)
        
        with torch.no_grad():
            sentiment = self.sentiment_pipeline(answer[:500])[0]
        
        return {
            "transcript": transcript,
            "answer": answer,
            "sentiment": sentiment
        }

processor = VideoProcessor()

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