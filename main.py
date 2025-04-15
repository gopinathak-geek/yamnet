import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd
import numpy as np
import os
import time
from notification import send_telegram_message
from wiam import classify_cry
# --- CONFIGURATIONS ---
MODEL_PATH = 'models/yamnet'
AUDIO_FILE = 'demo.m4a'
CHUNK_DURATION = 1  # in seconds
SR = 16000
CHUNK_SIZE = SR * CHUNK_DURATION


ALERT_THRESHOLD = 1  # how many consecutive "baby cry" chunks before alerting
COOLDOWN_PERIOD = 5  # seconds of silence before next alert can be sent

# --- LOAD MODEL + AUDIO ---
model = hub.load(MODEL_PATH)
label_path = os.path.join(MODEL_PATH, "assets", "yamnet_class_map.csv")
class_names = pd.read_csv(label_path)['display_name'].tolist()

waveform, sr = librosa.load(AUDIO_FILE, sr=SR)
num_chunks = len(waveform) // CHUNK_SIZE
print(f"📦 Total audio: {len(waveform)/SR:.2f} sec → {num_chunks} chunks")

# --- STATE VARIABLES ---
cry_counter = 0
cry_detected = False
last_cry_timestamp = None
simulation_start_time = time.time()

# --- MAIN LOOP ---
for i in range(num_chunks):
    chunk = waveform[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
    current_time = time.time()
    elapsed = int(current_time - simulation_start_time)

    # YAMNet inference
    scores, _, _ = model(chunk)
    mean_scores = tf.reduce_mean(scores, axis=0)
    top5 = tf.argsort(mean_scores, direction='DESCENDING')[:5]

    print(f"\n🧠 Chunk {i+1}/{num_chunks} — Time {elapsed}s")
    for idx in top5:
        print(f"{class_names[idx]}: {mean_scores[idx].numpy():.4f}")

    is_cry = any("baby cry" in class_names[idx].lower() and mean_scores[idx] > 0.5 for idx in top5)

    # Detection logic
    if is_cry:
        cry_counter += 1
        print(f"🍼 Cry Counter: {cry_counter}")
        if not cry_detected and cry_counter >= ALERT_THRESHOLD:
            cry_detected = True
            last_cry_timestamp = elapsed
            print(f"🚨 Baby crying detected at {elapsed}s — Sending alert!")
            reason = classify_cry(chunk)
            send_telegram_message(f"🚨 Baby started crying reason may be {reason}")
        elif cry_detected:
            last_cry_timestamp = elapsed  # update last known cry
    else:
        if cry_detected:
            print("⏳ Waiting for cooldown...")
            if elapsed - last_cry_timestamp > COOLDOWN_PERIOD:
                print(f"✅ Cry session ended at {elapsed}s — Resetting state.")
                cry_detected = False
                cry_counter = 0
        else:
            cry_counter = max(0, cry_counter - 1)  # smooth decay for spurious noise

