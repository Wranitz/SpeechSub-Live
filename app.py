from flask import Flask, request, jsonify, render_template
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import audioread
import librosa
import os

app = Flask(__name__)

#Load the model and processor
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
processor = Wav2Vec2Processor.from_pretrained(model_name) 
model = Wav2Vec2ForCTC.from_pretrained(model_name)

#make a home page pointer on '/' as index.html
@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')



@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio'] 

    #define a permanent file path
    audio_path = "uploaded_audio_chunk.wav"

    try:
        #save the audio file
        audio_file.save(audio_path)
        if os.path.getsize(audio_path) == 0:
            raise Exception("Received empty audio file chunk")
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return jsonify({'error': 'Failed to save audio file'})
    
     # Read the audio file using audioread
    try: 
        with audioread.audio_open(audio_path) as f: 
            sr = f.samplerate 
            channels = f.channels 
            print(f"Sample rate:{sr}, Channels:{channels}")
            
            audio_array = [] 
            for buf in f: 
                audio_array.extend(np.frombuffer(buf, dtype=np.int16)) 
    except Exception as e:
        print(f"Error reading audio file:{e}")
        return jsonify({'error': 'Failed to read audio file'})
    

    try:       
        audio_array = np.array(audio_array, dtype=np.float32)

        #if audio is two channel change it to 1 channel
        if channels >1:
            audio_array = audio_array.reshape((-1,channels)).mean(axis=1)

        #if audio is not 16000hz change it to 16khz
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        input_values = processor(audio_array, return_tensors="pt", sampling_rate=sr).input_values 
        logits = model(input_values).logits 
        predicted_ids = torch.argmax(logits, dim=-1) 
        transcription = processor.decode(predicted_ids[0]) 
    except Exception as e:
        print(f"Error Loading Model: {e}")
        return jsonify({'error' : 'Failed to Loading Model'})
    finally:
        # Delete the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return jsonify({'transcription': transcription})

if __name__ == "__main__":
    app.run(debug=True)