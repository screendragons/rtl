import re
import os
import whisper
from datasets import Audio, Dataset, load_dataset, load_metric
import string
import AudioFile

class Ort:
    def to_string(path):
        with open(path, "r", encoding = "ISO-8859-1") as f:
            filtered_lines = []
            for line in f:
                if re.search(r'"[a-zA-Z0-9\.\s]*"', line):
                    filtered_lines.append(line.strip())


            # Remove the first 6 lines
            filtered_lines = filtered_lines[6:]
            # Remove the last 4 lines
            # filtered_lines = filtered_lines[:4]

            all_text = ""

            # Print the filtered text
            for line in filtered_lines:
                # Skip string if empty
                if line == '""':
                    continue

                # add a space after each line
                line += " "

                all_text += line

            # Remove all double quotes from the text
            all_text = all_text.replace('"', ''      )

             # Remove all punctuation from the text
            all_text = all_text.translate(str.maketrans("", "", string.punctuation))


            return all_text


def run_model(model_name):
    model = whisper.load_model(model_name)
    audiofile = os.getcwd() + "/files/recordings/fn000151.wav"
    print("Transcribing " + audiofile + " with " + model_name + " model")
    return model.transcribe(audiofile, fp16=False)


def evaluate(result, model_name, ort_transcription):
    # Converting target to lowercase
    # target = [ort_transcription.lower()]  

    # Converting prediction to lowercase
    # prediction = [result['text'].lower()]  

     # Removing all punctuation from the reference and prediction
    target = [ort_transcription.lower().translate(str.maketrans("", "", string.punctuation))]
    prediction = [result['text'].lower().translate(str.maketrans("", "", string.punctuation))]



    # target = [ort_transcription]
    # prediction = [result['text']]

    # load metric
    wer = load_metric("wer")
    cer = load_metric("cer")

    # compute metrics
    wer_result = wer.compute(references=target, predictions=prediction)
    cer_result = cer.compute(references=target, predictions=prediction)

    print("results for " + model_name + " model:")
    print("WER: " + str(wer_result))
    print("CER: " + str(cer_result))


ort_transcription = Ort.to_string("files/transcriptions/fn000151.ort")


# evaluate(run_model("tiny"), model_name="tiny", ort_transcription=ort_transcription)
# evaluate(run_model("base"), model_name="base", ort_transcription=ort_transcription)
# evaluate(run_model("small"), model_name="small", ort_transcription=ort_transcription)
# evaluate(run_model("medium"), model_name="medium", ort_transcription=ort_transcription)

def result_gradio():  
    result = evaluate(run_model("tiny"), model_name="large", ort_transcription=ort_transcription)
    return result