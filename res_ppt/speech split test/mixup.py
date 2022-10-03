import librosa
import numpy as np
import soundfile as sf
import os


if __name__ == '__main__':
    speech = librosa.load(r"speech\speech_test_10s.wav", sr=16000, mono=True, offset=0., duration=10)
    noiseFileList = [os.path.join("noise", fileName) for fileName in os.listdir("noise")]

    for i in range(len(noiseFileList)):
        noise = librosa.load(noiseFileList[i], sr=16000, mono=True, offset=0., duration=10)
        noisy = speech[0] + noise[0]
        sf.write(os.path.join("noisy", f"split_test{i}.wav"), noisy, samplerate=16000)

