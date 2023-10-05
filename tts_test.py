

# Text to speech requires gtts installed
# Languages gtts-cli --all
# https://gtts.readthedocs.io/en/latest/module.html#localized-accents
from gtts import gTTS 
# for audio play required playsound
from playsound import playsound

text= "test"
language= "en"

print('-OK, Generating audio...')
speech = gTTS(text = text, lang = language, tld='fr', slow = False)

print('-OK, Saving audio...')
path='tts.mp3'
speech.save(path)

print('-OK, Play audio...')
playsound(path)