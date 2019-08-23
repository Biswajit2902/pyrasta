import librosa

def wavread(wavfilepath, fs=16000):
    try:
        y,sr = librosa.load(wavfilepath, fs)
        return y, sr
    except Exception as e:
        raise e

def wavwrite(wavfilepath, y, fs):
    try:
        scipy.io.wavfile.write(wavfilepath, fs, y)
    except Exception as e:
        raise e
