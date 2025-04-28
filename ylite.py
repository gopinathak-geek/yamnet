from tensorflow.lite.python.interpreter import Interpreter  #macos
import numpy as np
import zipfile
import librosa  
import sounddevice as sd

SR = 16000
CHUNK_DURATION = 0.975
CHUNK_SIZE = int(SR * CHUNK_DURATION)
MODEL_PATH = '/Users/gopinathak/myspace/projects/yamnet/models/1.tflite'

interpreter = Interpreter(MODEL_PATH)

input_details = interpreter.get_input_details()
waveform_input_index = input_details[0]['index']
output_details = interpreter.get_output_details()
scores_output_index = output_details[0]['index']



while True:
    try:
        audio = sd.rec(CHUNK_SIZE, samplerate=SR, channels=1, dtype='float32')
        sd.wait()
        chunk = audio.flatten()

        interpreter.resize_tensor_input(waveform_input_index, [chunk.size], strict=True)
        interpreter.allocate_tensors()
        interpreter.set_tensor(waveform_input_index, chunk)
        interpreter.invoke()
        scores = interpreter.get_tensor(scores_output_index)
        
        labels_file = zipfile.ZipFile(MODEL_PATH).open('yamnet_label_list.txt')
        labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]
        
        top5_indices = np.argsort(scores[0])[-5:][::-1]  # Sort descending
        for idx in top5_indices:
            print(f"{labels[idx]} (score: {scores[0][idx]:.3f})")

        # top_class_index = scores.argmax()
        # labels_file = zipfile.ZipFile(MODEL_PATH).open('yamnet_label_list.txt')
        # labels = [l.decode('utf-8').strip() for l in labels_file.readlines()]
        # print(labels[top_class_index])  # Should print 'Silence'.
    
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
        break


# interpreter.resize_tensor_input(waveform_input_index, [waveform.size], strict=True)
# interpreter.allocate_tensors()
# interpreter.set_tensor(waveform_input_index, waveform)
# interpreter.invoke()
# scores = interpreter.get_tensor(scores_output_index)
# print(scores.shape)  # Should print (1, 521)

