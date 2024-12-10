from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from pydub import AudioSegment
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = '78HWEOP7545AITE'
socketio = SocketIO(app)

# Load the model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

@socketio.on('start')
def handle_start():
    emit('status', {'status': 'Recording started'})

@socketio.on('stop')
def handle_stop():
    emit('status', {'status': 'Recording stopped'})

@socketio.on('audio_data')
def handle_audio(data):
    audio_bytes = BytesIO(data)
    audio_segment = AudioSegment.from_file(audio_bytes, format="wav")
    samples = audio_segment.get_array_of_samples()
    waveform = torch.tensor(samples).float() / 32768.0  # Normalize to [-1, 1]
    
    input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    emit('transcription', {'transcription': transcription})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, debug=True)
