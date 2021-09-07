import os
import time
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from spectralcluster import SpectralClusterer
from resemblyzer import sampling_rate

#give the file path to your audio file
audio_file_path = 'test_samples/speaker_identification_testing/9986383563_sheik_ienergizer@olacabs.com_2021-08-01-14-44-43.wav'

def convert(seconds):
    try:
        milli=str(seconds).split('.')[-1][:2]
    except:
        milli='00'
    return time.strftime(f"%H:%M:%S,{milli}", time.gmtime(seconds))

def create_labelling(labels,wav_splits):
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    labelling = []
    start_time = 0

    for i,time in enumerate(times):
        if i>0 and labels[i]!=labels[i-1]:
            temp = [str(labels[i-1]),start_time,time]
            labelling.append(tuple(temp))
            start_time = time
        if i==len(times)-1:
            temp = [str(labels[i]),start_time,time]
            labelling.append(tuple(temp))

    return labelling

def main():
    wav_fpath = Path(audio_file_path)
    wav = preprocess_wav(wav_fpath)
    # print(wav.shape)
    encoder = VoiceEncoder("cpu")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=8)
    # print(cont_embeds.shape)

    clusterer = SpectralClusterer(
        min_clusters=2,
        max_clusters=100,
        p_percentile=0.90,
        gaussian_blur_sigma=1)

    labels = clusterer.predict(cont_embeds)
    labelling = create_labelling(labels,wav_splits)
    
    output_file = str(os.path.splitext(audio_file_path)[0]) + ".txt"
    with open(output_file, 'w') as f:
        for item in labelling:
            f.write("{} : {} -> {} \n".format(convert(item[1]),convert(item[2]),item[0]))

if __name__ == "__main__":
    main()