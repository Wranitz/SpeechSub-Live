from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')

if __name__ == "__main__"