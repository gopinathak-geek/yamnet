from utils import load_config
import tensorflow_hub as hub
import os
import pandas as pd
import sounddevice as sd
import tensorflow as tf

config = load_config('config.json')
chunk_size = config.sr * config.chunk_duration
cry_counter = 0
cry_event = False  # To avoid duplicate alerts

model = hub.load(config.model_path)
label_path = os.path.join(config.model_path, "assets", "yamnet_class_map.csv")
class_names = pd.read_csv(label_path)['display_name'].tolist()

while True:
    try:
        audio = sd.rec(chunk_size, samplerate=config.sr, channels=1, dtype='float32')
        sd.wait()
        input = audio.flatten()
        scores, embeddings, spectrogram = model(input)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        cry_detected = any(mean_scores[i] > 0.5 for i in config.cry_indices)

    
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
        break