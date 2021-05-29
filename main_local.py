import os
import secrets
from flask import Flask, render_template, request, flash, redirect, send_from_directory
from werkzeug.utils import secure_filename
import torch
import torchaudio
from model import model, transform, channel, label_encoder


UPLOAD_FOLDER = "static/upload"
ALLOWED_EXTENSIONS = {"mp3"}
SAVED_NAME = "tmp.mp3"
TEN_SECONDS_FRAMES = 443520


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.secret_key = secrets.token_urlsafe(32)


@app.route("/")
def hello():
    """simple landing Page."""
    return render_template("index.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("ERROR: No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("ERROR: No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            pth = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(pth)
            return render_template(
                "index.html", audio_fname=filename, qari=predict(pth)
            )
    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def app_handle_413(e):
    return render_template("index.html", qari="File is too Large")


def predict(path):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.size()[1] < TEN_SECONDS_FRAMES:
        return "Audio File is too short"

    sound_data = transform(waveform[channel, :TEN_SECONDS_FRAMES])[0:13, :].unsqueeze(0)

    model.eval()

    with torch.no_grad():
        output = model(sound_data)

    qari = label_encoder.inverse_transform([torch.argmax(output).item()])[0]

    return qari


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
