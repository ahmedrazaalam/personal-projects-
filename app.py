from flask import Flask, render_template, request, session, g
import numpy as np
import sqlite3
import pandas as pd
import librosa

app = Flask(__name__)
app.secret_key = "KjhLJF54f6ds234H"

DATABASE = "mydb.sqlite3"

dataset = pd.read_csv(r'Audio-Deepfake-Detection-main/dataset.csv')

num_mfcc = 100
num_mels = 128
num_chroma = 50

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

@app.route('/about.html')
def about():
    background_image = "/static/image2.jpg"
    return render_template('about.html', background_image=background_image)

@app.route('/index.html')
def home1():
    background_image = "/static/image1.jpg"
    return render_template('index.html', background_image=background_image)

@app.route('/model.html', methods=['GET', 'POST'])
def model():
    background_image = "/static/image5.jpg"
    loader_visible = False

    if request.method == 'POST':
        # print(request)
        selected_file = request.files['audio_file']
        file_name = selected_file.filename

        file_stream = selected_file.stream
        file_stream.seek(0)

        X, sample_rate = librosa.load(file_stream, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
        chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
        flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)
        features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
        distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
        closest_match_idx = np.argmin(distances)
        closest_match_label = dataset.iloc[closest_match_idx, -1]
        total_distance = np.sum(distances)
        closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
        closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

        if closest_match_label == 'deepfake':
            file_label = f"File: {file_name}"
            result_label = f"Result: Fake with {closest_match_prob_percentage}%"
        else:
            file_label = f"File: {file_name}"
            result_label = f"Result: Real with {closest_match_prob_percentage}%"

        return render_template('model.html', file_label=file_label, result_label=result_label, background_image=background_image,loader_visible=loader_visible)
    else:
        return render_template('model.html', background_image=background_image,loader_visible=loader_visible)


if __name__ == "__main__":
     app.run(debug=True)
