from flask import Flask, request, jsonify
import torch
from transformers import pipeline
import whisper

app = Flask(__name__)

# Charger Whisper large pour transcription
model_whisper = whisper.load_model("large")

# Charger BART large CNN pour résumé
summarizer = pipeline("summarization", model="sshleifer/bart-large-cnn")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    file = request.files["audio"]
    file.save("temp_audio.wav")
    result = model_whisper.transcribe("temp_audio.wav")
    return jsonify({"transcription": result["text"]})

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.json["text"]
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    return jsonify({"summary": summary[0]['summary_text']})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
