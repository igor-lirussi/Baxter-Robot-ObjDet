import torch
import whisper
import queue
import speech_recognition as sr
import threading
import numpy as np
import os
import tempfile
import platform
import pynput.keyboard

from whisper_mic.utils import get_logger


class WhisperMicIgor:
    """
    energy: (energy threshold) Represents the energy level threshold for sounds. Values below this threshold are considered silence, and values above this threshold are considered speech. (if you wanna see it in real time you need to change file in conda->python->site-packages->speech recognition->__init__.py)
    pause: (pause threshold) Represents the minimum length of silence (in seconds) that will register as the end of a phrase. Can be changed. Smaller values result in the recognition completing more quickly, but might result in slower speakers being cut off. 
    timeout: the maximum number of seconds that this will wait for a phrase to START before giving up and throwing an speech_recognition.WaitTimeoutError exception. If timeout is None, there will be no wait timeout.
    phrase_time_limit: The phrase_time_limit parameter is the maximum number of seconds that this will allow a phrase to continue after starting before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If phrase_timeout is None, there will be no phrase time limit.
    """
        
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),english=False,verbose=False,energy=300,pause=2,initial_energy_adjust=False,dynamic_energy=False,save_file=False, model_root="~/.cache/whisper",mic_index=None):
        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.pause = pause
        self.initial_energy_adjust = initial_energy_adjust
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english
        self.keyboard = pynput.keyboard.Controller()

        self.platform = platform.system()

        if self.platform == "darwin":
            if device == "mps":
                self.logger.warning("Using MPS for Mac, this does not work but may in the future")
                device = "mps"
                device = torch.device(device)

        if (model != "large" and model != "large-v2") and self.english:
            model = model + ".en"
        
        self.audio_model = whisper.load_model(model, download_root=model_root).to(device)
        self.temp_dir = tempfile.mkdtemp() if save_file else None

        self.audio_queue = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()
        
        self.break_threads = False
        self.mic_active = False

        self.banned_results = [""," ","\n",None]

        self.__setup_mic(mic_index)


    def __setup_mic(self, mic_index):
        if mic_index is None:
            self.logger.info("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy
        if self.initial_energy_adjust:
            with self.source:
                    print("Please wait. Calibrating microphone...")   
                    # listen for 5 seconds and calculate the ambient noise energy level   
                    self.recorder.adjust_for_ambient_noise(self.source, duration=5)
        print("Sound energy threshold "+str(self.recorder.energy_threshold))
        self.logger.info("Mic setup complete")


    def __preprocess(self, data):
        return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)

    
    def __get_all_audio(self):
        audio = bytes()
        got_audio = False
        while not got_audio:
            while not self.audio_queue.empty():
                audio += self.audio_queue.get()
                got_audio = True

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data
    

    # Handles the task of getting the audio input via microphone. This method has been used both for listen() method and for the __transcribe_forever() method (which has been used for implementing the listen_loop() method)
    def __listen_handler(self, timeout, phrase_time_limit):
        try:
            with self.source as microphone:
                audio = self.recorder.listen(source=microphone, timeout=timeout, phrase_time_limit=phrase_time_limit)
            self.__record_load(audio)
            audio_data = self.__get_all_audio()
            self.__transcribe(data=audio_data)
        except sr.WaitTimeoutError:
            self.result_queue.put_nowait("Timeout: No speech detected within the specified time.")
        except sr.UnknownValueError:
            self.result_queue.put_nowait("Speech recognition could not understand audio.")


    # This method is similar to the __listen_handler() method but it has the added ability for recording the audio for a specified duration of time
    def __record_handler(self, duration, offset):
        with self.source as microphone:
            audio = self.recorder.record(source=microphone, duration=duration, offset=offset)
        
        self.__record_load(audio)
        audio_data = self.__get_all_audio()
        self.__transcribe(data=audio_data)


    # This method takes the recorded audio data, converts it into raw format and stores it in a queue. 
    def __record_load(self, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)


    def __transcribe_forever(self,timeout, phrase_time_limit) -> None:
        while True:
            if self.break_threads:
                break
            self.__listen_handler(timeout, phrase_time_limit)


    def __transcribe(self,data=None, realtime: bool = False) -> None:
        if data is None:
            audio_data = self.__get_all_audio()
        else:
            audio_data = data
        audio_data = self.__preprocess(audio_data)
        if self.english:
            result = self.audio_model.transcribe(audio_data,language='english')
        else:
            result = self.audio_model.transcribe(audio_data)

        predicted_text = result["text"]
        if not self.verbose:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(predicted_text)
        else:
            if predicted_text not in self.banned_results:
                self.result_queue.put_nowait(result)

        if self.save_file:
            os.remove(audio_data)


    def listen_loop(self, dictate: bool = False, timeout=None, phrase_time_limit=None) -> None:
        self.logger.info("Listening...")
        threading.Thread(target=self.__transcribe_forever, args=(timeout, phrase_time_limit)).start()
        while True:
            result = self.result_queue.get()
            if dictate:
                self.keyboard.type(result)
            else:
                print(result)

            
    def listen(self, timeout = None, phrase_time_limit=None):
        self.logger.info("Listening...")
        self.__listen_handler(timeout, phrase_time_limit)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()


    # This method is similar to the listen() method, but it has the ability to listen for a specified duration, mentioned in the "duration" parameter.
    def record(self, duration=None, offset=None):
        self.logger.info("Listening...")
        self.__record_handler(duration, offset)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()


    def toggle_microphone(self) -> None:
        #TO DO: make this work
        self.mic_active = not self.mic_active
        if self.mic_active:
            print("Mic on")
        else:
            print("turning off mic")
            self.mic_thread.join()
            print("Mic off")