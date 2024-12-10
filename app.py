from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import io
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import threading

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

def handle_audio_stream(data): 
    try: 
        audio_bytes = io.BytesIO(data) 
    except Exception as e: 
        print(f"Error converting data to BytesIO: {e}") 
        return 
    
    try: 
        audio_segment = AudioSegment.from_file(audio_bytes, format='webm') 
    except Exception as e: 
        print(f"Error reading audio segment from BytesIO: {e}") 
        return 
    
    try: 
        waveform = np.array(audio_segment.get_array_of_samples()) 
    except Exception as e: 
        print(f"Error converting audio segment to numpy array: {e}") 
        return 
    
    try: 
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values 
    except Exception as e: 
        print(f"Error processing audio waveform: {e}") 
        return 
    
    try: 
        logits = model(input_values).logits 
        predicted_ids = torch.argmax(logits, dim=-1) 
        transcription = processor.decode(predicted_ids[0]) 
    except Exception as e: 
        print(f"Error transcribing audio: {e}") return 
    
    
    try: 
        socketio.emit('transcription', {'transcription': transcription}) 
    except Exception as e: 
        print(f"Error emitting transcription to client: {e}")

@socketio.on('start')
def handle_start():
    emit('status', {'status': 'Recording started'})

@socketio.on('stop')
def handle_stop():
    emit('status', {'status': 'Recording stopped'})

@socketio.on('audio_data')
def handle_audio(data):
    handle_audio_stream(data)

if __name__ == "__main__":
    socketio.run(app, debug=True)
