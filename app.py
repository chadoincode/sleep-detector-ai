from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)  # biar frontend Astro bisa akses

# Load model dari file
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])

def predict():
    data = request.get_json()
    print("DATA MASUK:", data)

    try:
        # print("Tipe data masing-masing:")
        # for key, val in data.items():
        #     print(f"{key}: {val} ({type(val)})")
        # Load model & scaler
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        features = [
            int(data["age"]),
            float(data["sleep_duration"]),
            int(data["stress_level"]),
            int(data["physical_activity_level"]),
            int(data["heart_rate"]),
            int(data["gender"])
        ]

        input_array = scaler.transform([features])
        prediction = model.predict(input_array)

        hasil = "Kualitas Tidur Buruk" if prediction[0] == 0 else "Kualitas Tidur Baik"
        return jsonify({"hasil": hasil})

    except Exception as e:
        print("ERROR SAAT PREDIKSI:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default ke 5000 kalau gak ada PORT
    app.run(host="0.0.0.0", port=port)
