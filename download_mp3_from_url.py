import os
import requests

url = "https://media.talkbank.org/ca/CallHome/eng/4604.mp3"
output_dir = "test_samples"
file_name = "audio_4604.mp3"

doc = requests.get(url)

with open(os.path.join(output_dir,file_name), 'wb') as f:
        f.write(doc.content)