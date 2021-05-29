import os
import secrets
from flask import Flask, render_template, request, jsonify, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename

from pydub import AudioSegment


UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'mp3'}
SAVED_NAME = 'tmp.mp3'
TEN_SECONDS = 10045

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
            audio.export
            return render_template("index.html", audio_fname=filename)
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # audio = AudioSegment.from_mp3(src)
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/predict", methods=["POST"])
def predict():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
