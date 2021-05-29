import os
import secrets
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename

from pydub import AudioSegment
import torch
import torchaudio

from model import model, transform, channel

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'mp3'}
SAVED_NAME = 'tmp.mp3'
TEN_SECONDS = 10*1000

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

app.secret_key = secrets.token_urlsafe(32)

@app.route("/")
def hello():
    """simple landing Page."""
    return render_template("index.html")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('ERROR: No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('ERROR: No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pth = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(pth)
            audio = AudioSegment.from_mp3(pth)
            audio[:TEN_SECONDS].export(pth, format="mp3")
            return render_template("index.html", audio_fname=filename, qari=predict(pth))
    return render_template("index.html")


def predict(path):
    waveform, sample_rate = torchaudio.load(path)
    sound_data = transform(waveform[channel, :])[0:13, :].unsqueeze(0)

    model.eval()
    
    with torch.no_grad():
        output = model(sound_data)

    qari = label_encoder.inverse_transform([torch.argmax(output).item()])[0]
        
    return qari


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
