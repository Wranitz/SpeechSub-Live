from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    transcription = "Hello World"
    return jsonify({'transcription': transcription})


if __name__ == "__main__":
    app.run(debug=True)