from whisper_mic_igor import WhisperMicIgor #requires a conda env with whisper-mic installed and opencv (opencv-python) and ros (rospkg) 
# https://github.com/mallorbc/whisper_mic/
# if you have problems with wrong libffi (since opencv and ros conflics) conda install libffi==3.3
# do not call your code "whisper.py" will create circular dependency
# if you have a lot of ALSA messages sudo nano /usr/share/alsa/alsa.conf (https://github.com/Uberi/speech_recognition/issues/526) (https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time)
# or set this handler to asound library
from ctypes import *
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
  #print('error received')
  pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
# Set error handler of asound libray to the previous one
asound.snd_lib_error_set_handler(c_error_handler)

'''
# to see your anbient noise, I suggest to keep the threshold 200 above it to avoid false positive speech
import speech_recognition as sr   
r = sr.Recognizer()   
with sr.Microphone() as source:   
    print("Please wait. Calibrating microphone...")   
    # listen for 5 seconds and calculate the ambient noise energy level   
    r.adjust_for_ambient_noise(source, duration=5)
    print("Sound energy threshold "+str(r.energy_threshold))
'''

# create mic module
mic = WhisperMicIgor(model="base",english=True,verbose=False,energy=900,pause=1.8,dynamic_energy=False,save_file=False, model_root="~/.cache/whisper",mic_index=None) #in the console whisper_mic --help for info on usage

# code execution
import time
print('sleeping for a while to simluate code execution')
time.sleep(10)

print("START")
result = mic.listen(phrase_time_limit=20)
print(result)
print("STOP")