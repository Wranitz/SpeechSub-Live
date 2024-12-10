from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import io
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = Flask(__name__)
app.config['SECRET_KEY'] = '78HWEOP7545AITE'
socketio = SocketIO(app)

# Load the model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start')
def handle_start():
    emit('status', {'status': 'Recording started'})

@socketio.on('stop')
def handle_stop():
    emit('status', {'status': 'Recording stopped'})

@socketio.on('audio_data')
def handle_audio(data):
    audio_bytes = io.BytesIO(data)
    audio_segment = AudioSegment.from_file(audio_bytes, format='webm')
    waveform = np.array(audio_segment.get_array_of_samples())
    
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    emit('transcription', {'transcription': transcription})

if __name__ == "__main__":
    socketio.run(app, debug=True)
