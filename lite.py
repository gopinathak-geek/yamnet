import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
import time

# === CONFIG ===
SR = 16000
CHUNK_DURATION = 1  # seconds
CHUNK_SIZE = SR * CHUNK_DURATION
MODEL_PATH = 'models/yamnet'  # TFHub yamnet
cry_event = False
last_cry_time = None
CRY_COOLDOWN = 15

# === Load model and labels once ===
model = hub.load(MODEL_PATH)
label_path = os.path.join(MODEL_PATH, "assets", "yamnet_class_map.csv")
class_names = pd.read_csv(label_path)['display_name'].tolist()

print("🎤 Listening... Press Ctrl+C to stop")

while True:
    try:
        audio = sd.rec(CHUNK_SIZE, samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        chunk = audio.flatten()

        # Step 2: Run inference
        scores, embeddings, spectrogram = model(chunk)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        # Step 3: Get top prediction
        top_idx = np.argmax(mean_scores)
        top_label = class_names[top_idx]
        top_score = mean_scores[top_idx]

        print(f"🧠 {top_label} ({top_score:.2f})")
        current_time = time.time()

        if "cough" in top_label.lower() and top_score > 0.5:
            last_cry_time = current_time
            if not cry_event:
                cry_event = True
                print("🚨 Baby cry event started!")
        else:
            if cry_event and last_cry_time and (current_time - last_cry_time > CRY_COOLDOWN):
                cry_event = False
                print("✅ Baby cry event ended (15s silence)")

    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
        break
