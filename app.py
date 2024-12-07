from flask import Flask, request, jsonify, render_template
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
from pydub import AudioSegment

app = Flask(__name__)

#Load the model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name) 
model = Wav2Vec2ForCTC.from_pretrained(model_name)


@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')



@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio'] 
    # Read the audio file using pydub 
    audio = AudioSegment.from_file(audio_file) 
    audio_data = np.array(audio.get_array_of_samples()) 
    samplerate = audio.frame_rate 
    input_values = processor(audio_data, return_tensors="pt", sampling_rate=samplerate).input_values 
    logits = model(input_values).logits 
    predicted_ids = torch.argmax(logits, dim=-1) 
    transcription = processor.decode(predicted_ids[0]) 
    return jsonify({'transcription': transcription})

if __name__ == "__main__":
    app.run(debug=True)