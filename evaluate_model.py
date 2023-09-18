import re
import numpy as np 
import matplotlib.pyplot as plt 
import whisper
import os
import pandas as pd
from Ort import Ort
from AudioFile import AudioFile
from datasets import load_metric
import string

# Set the path to the audio files located in the files/recordings folder
#AUDIO_FILES_PATH = os.getcwd() + "/files/recordings/"
cwd = os.getcwd()
AUDIO_FILES_PATH = os.path.join(cwd, "files/recordings/", "")

# Metadata file path
METADATA_FILE_PATH = os.getcwd() + "/files/metadata/recordings_NL.txt"

# The path to the ORT files containing the transcriptions
ORT_TRANSCRIPTIONS_PATH = os.getcwd() + "/files/transcriptions/"

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline

model = WhisperForConditionalGeneration.from_pretrained("hannatoenbreker/whisper-dutch")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Dutch", task="transcribe")
pipe = pipeline(model="openai/whisper-small", chunk_length_s=30)

# Create the method for running the model
def run_model(audio_file_path):
    print("Transcribing " + audio_file_path)
    return transcribe(audio_file_path)

def transcribe(audio):
    text = pipe(audio, generate_kwargs = {"language":"<|nl|>", "task": "transcribe"})
    return text

def get_error_rate(group_nr):
    results = {}
    data = []

    # Get the audio files for the given subgroup
    audio_files = AudioFile.get_audio_files_by_subgroup(METADATA_FILE_PATH, AUDIO_FILES_PATH, component="comp-q", subgroup=group_nr)
    print("there are " + str(len(audio_files)) + " audio files present.")

    # Run the model on all audio files
    for audio_file in audio_files:
        # Get the file name without the extension so we can find the corresponding ORT file
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
    
        # Run the model on the audio file
        result = run_model(audio_file)

        # append the result to the results dictionary with the file name as key
        results[file_name] = result

    wer = load_metric("wer")
    cer = load_metric("cer")

    # Loop trough the results dictionary
    for result in results.items():
        # Set magic numbers
        FILE_NAME_INDEX = 0
        TRANSCRIPTION_INDEX = 1

        # Get the path to the ORT file
        ort_file_path = os.getcwd() + "/files/transcriptions/" + result[FILE_NAME_INDEX] + ".ort"

        # Converting the ort transcription to a string and transform it to lowercase
        target = [(Ort.to_string(ort_file_path)).lower()]

        # Define the prediction, convert it to lowercase and remove punctuation
        prediction = [result[TRANSCRIPTION_INDEX]["text"].lower().translate(str.maketrans("", "", string.punctuation))]

        # compute metrics
        wer_result = wer.compute(references=target, predictions=prediction)
        cer_result = cer.compute(references=target, predictions=prediction)

        metadata = AudioFile.get_metadata_by_nr(result[FILE_NAME_INDEX])

        data.append({'wer': wer_result, 'cer': cer_result, 'subgroup': group_nr, 'gender': metadata['Gender'].iloc[0]})

    return pd.DataFrame(data)

subgroup_1 = get_error_rate(1)
subgroup_2 = get_error_rate(2)

data = pd.concat([subgroup_1, subgroup_2], ignore_index=True)
print(data)