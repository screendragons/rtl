import os
import torch
import sys
import transformers
import time
import librosa
import re
import pandas as pd
import unidecode

from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq

import whisper

from pyctcdecode import BeamSearchDecoderCTC
from jiwer import wer
from jiwer import cer

from abc import ABC, abstractmethod
import torch

class asr_model(ABC):
    @abstractmethod
    def predict(self, file):
        pass

    def normalise_text(self, text: str) -> str:
       chars_to_ignore_regex = '[\',?.!\-\;\:"“%‘”�—’…–]'
       text = re.sub(chars_to_ignore_regex, "", text.lower())
       return text
    

class whisper_asr_model(asr_model):
    def __init__(self, lm):
        #torch.cuda.set_device(1)
        print('whisper constructor running')
        self.model = whisper.load_model(lm)

    def predict(self, file):
        prediction = self.model.transcribe(file, language='nl')
        prediction['text'] = self.normalise_text(prediction['text'])
        return prediction
    
class rtl_whisper_asr_model(asr_model):
    def __init__(self):
        # torch.cuda.set_device(0)
        print('whisper constructor running')

    def predict(self, file):
        pipe = pipeline(model="hannatoenbreker/whisper-dutch-small-v2",
                        chunk_length_s=10, stride_length_s=(4, 2),
                        device="cuda:0",
                        generate_kwargs = {"language":"<|nl|>","task": "transcribe"})
        
        prediction = pipe(file)
        prediction['text'] = self.normalise_text(prediction['text'])
        return prediction
    
class synthetic_whisper_asr_model(asr_model):
    def __init__(self):
        #torch.cuda.set_device(1)
        print('whisper constructor running')

    def predict(self, file):
        pipe = pipeline(model= "LisanneH/whisper-small-nl-Synthetic_2", 
                        chunk_length_s=30,
                        device="cuda:1")
        prediction = pipe(file)
        prediction['text'] = self.normalise_text(prediction['text'])
        return prediction

class wav2vec2_asr_model(asr_model):

    def __init__(self):
        print('wav2vec constructor running')

        return # remove when desired

        # model id
        model_id = '/home/' + os.environ['USER'] + '/AAI_Data/AAI_project2/ASR_project/models/xls-r-2b-nl-v2_lm-5gram-os'

        # individually initializing all the components of speech recognition
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        config = transformers.PretrainedConfig.from_pretrained(model_id)
        model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = transformers.AutoProcessor.from_pretrained(model_id)
        language_model = BeamSearchDecoderCTC.model_container[processor.decoder._model_key]._kenlm_model
    
        # Running on GPU if pytorch detects that you have it, otherwise CPU (-1)
        device = 0 if torch.cuda.is_available() else -1 
        
        # 16KHz
        self.sampling_rate = 16000 

        # initializing the pipeline
        self.asr = pipeline("automatic-speech-recognition", \
                    config=config, model=model, tokenizer=tokenizer, \
                        feature_extractor=feature_extractor, decoder=processor.decoder, \
                            device=device, return_timestamps="word")

    def predict(self, file):
        print('running predict for file {}'.format(file))
        sound_array, _ = librosa.load(file, sr = self.sampling_rate) 
        prediction = self.map_to_pred(sound_array)
        prediction['text'] = self.normalise_text(prediction['text'])
        return prediction

    # map function to decode audio
    def map_to_pred(self, sound_array, chunk_length_s = 8, stride_length_s = 2):
        return self.asr(sound_array, chunk_length_s = chunk_length_s, stride_length_s = stride_length_s)
    
def initialise_model(asr_model):

    print("initialising model")

    if asr_model == 'wav2vec2':
        print('@initialise_model for wav2vec')
        asr = wav2vec2_asr_model()
    elif asr_model.startswith('whisper-'):
        print("Initialise model for standard whisper")
        asr = whisper_asr_model(asr_model.split('whisper-')[1])
    elif asr_model.startswith('rtl'):
        print('Initialise model for RTL model')
        asr = rtl_whisper_asr_model()
    elif asr_model.startswith('synthetic-'):
        print('Initialise model using synthetic data')
        asr = synthetic_whisper_asr_model()

    return asr

def get_reference_text(filename):
    with open(filename, encoding='ISO-8859-1') as f_read:
        content = f_read.read().splitlines()

        content_str = ''.join(f'{x.split(";")[2] + " "}' for x in content).strip()
        return content_str

def fix_words(text):
    substitutions = [("k", "ik"), \
                     ("t", "het"), \
                     ("s avonds", "savonds"), \
                     ("s morgens", "smorgens"), \
                     ("s middags", "smiddags"), \
                     ("s ochtends", "sochtends"), \
                     ("snachts", "s nachts"),
                     ("s", "is"), \
                     ("m", "hem"), \
                     ("r", "haar"), \
                     ("ns", "eens"), \
                     ("dr", "er"), \
                     ("drtussen", "ertussen"), \
                     ("mn", "mijn"), \
                     ("zn", "zijn"), \
                     ("das", "dat is"), \
                     ("zon", "zo een"), \
                     ("ie", "hij"), \
                     ("drbij", "doorbij"), \
                     ("uhm", ""), \
                     ("ehm", ""), \
                     ("n", "een"), \
                     ("oh", ""), \
                     ("eh", ""), \
                     ("ah", ""), \
                     ("he", "")]

    words = text.split()
    new_words = []
    for word in words:
        # Check if the word matches any of the source substitutions
        for source, target in substitutions:
            if word == source:
                word = target
                break
        new_words.append(word)
    return ' '.join(new_words)

def load_prediction_cache(asr_model, output_folder, comp, lang, version):
    print('loading prediction cache')

    cache = {}

    path = os.path.join(output_folder,'eval-' + asr_model + '-comp-' + comp + '-' + lang + '-log-' + version + '.txt')

    if os.path.exists(path):
        with open(path, encoding="ISO-8859-1") as f_read:
           content = f_read.read().splitlines()

        for i in range(0, len(content)):
            if i % 9 == 0:
               recording_id = content[i].split(' ')[0]

            if i % 9 == 6:
               prediction = content[i]
               cache[recording_id] = prediction

    return cache

def main(args):
    print(args)
    asr_model = args[1]
    audio_folder = args[2]
    reference_folder = args[3]
    meta_folder = args[4]
    comp = args[5]
    lang = args[6]
    output_folder = args[7]
    version = args[8]

    print('eval: asr={}, audio={}, ref={}, meta={}, c={}, l={}, out={} v={}'.\
            format(asr_model, audio_folder, reference_folder, meta_folder, comp, lang, output_folder, version))

    prediction_cache = load_prediction_cache(asr_model, output_folder, comp, lang, version)

    print('prediction cache size: {}'.format(len(prediction_cache)))

    st = time.time()
    asr = initialise_model(asr_model)
    #asr = None
    print('model loaded in {} seconds'.format(str(time.time() - st)))

    # loading recordings and speakers
    recordings_csv = pd.read_csv(os.path.join(meta_folder,lang,'recordings.txt'), sep='\t')
    speakers_csv = pd.read_csv(os.path.join(meta_folder,lang,'speakers.txt'), sep='\t')

    index = 0

    with open(os.path.join(output_folder,'eval-' + asr_model + '-comp-' + comp + '-' \
                + lang + '-results-' + version + '.txt'),'w', encoding="ISO-8859-1") as f_write:
       with open(os.path.join(output_folder,'eval-' + asr_model + '-comp-' + comp + '-' \
                + lang + '-log-' + version + '.txt'),'w', encoding="ISO-8859-1") as f_write_log:
           f_write.write("recording_id;speaker_id;component;group;" + \
                           "age;gender;cef;dialect_region;res_place;" + \
                           "birth_place;home_language_1;home_language_2;" + \
                           "comment;education;education_place;lengh_stay;time_dutch_l2;duration;" + \
                           "wer;cer;reference;prediction\n")
           for filename in os.listdir( os.path.join(audio_folder, 'comp-' + comp, lang)):

              #if filename != 'fn100167.wav' and \
              #   filename != 'fn000071_trim.wav' and \
              #   filename != 'fn000029_trim.wav' and \
              #   filename != 'fn000100.wav' and \
              #   filename != 'fn000067_trim.wav' and \
              #   filename != 'fn000757_trim.wav':
              #   continue

              if filename.endswith('.wav'):
                  index +=1 

                  print('processing recording {} - {}'.format(index,filename))

                  simplified_filename = re.split("[_trim.|.]", filename)[0]

                  st = time.time()

                  # check if prediction is cached - if yes, use it
                  prediction = {}
                  if prediction_cache.get(simplified_filename):
                      prediction['text'] = prediction_cache[simplified_filename]
                      print('woohoo cached for {}'.format(simplified_filename))
                  else:
                      prediction = asr.predict(os.path.join(audio_folder,'comp-' + comp, lang,filename))

                  print('\tprediction in {} seconds'.format(str(time.time() - st)))
                  #print('pred = {}'.format(prediction))

                  predicted_text = unidecode.unidecode(prediction['text'])
                  #predicted_text = 'hello'

                  #print('\tpredicted text={}'.format(predicted_text))

                  reference_text = get_reference_text(\
                      os.path.join( reference_folder, 'comp-'+ comp, lang, simplified_filename + '.ort.seg.txt'))

                  #print('\treference text={}'.format(reference_text))

                  reference_text = fix_words(reference_text)
                  predicted_text = fix_words(predicted_text)

                  wer_result = wer(reference_text, predicted_text)
                  cer_result = cer(reference_text, predicted_text)

                  # additional_data
                  recordings_row = recordings_csv[recordings_csv['Root']==simplified_filename]
                  speaker_id = recordings_row['SpeakerID'].values[0]

                  speaker_row = speakers_csv[speakers_csv['RegionSpeaker']==speaker_id]

                  component = recordings_row['Component'].values[0]
                  group = recordings_row['Group'].values[0]
                  age = recordings_row['Age'].values[0]
                  gender = recordings_row['Gender'].values[0]
                  cef = recordings_row['CEF'].values[0]
                  dialect_region = recordings_row['DialectRegion'].values[0]
                  res_place = speaker_row['ResPlace'].values[0]
                  birth_place = speaker_row['BirthPlace'].values[0]
                  home_language_1 = speaker_row['HomeLanguage1'].values[0]
                  home_language_2 = speaker_row['HomeLanguage2'].values[0]
                  comment = speaker_row['Comment'].values[0]
                  education = speaker_row['EduLevel'].values[0]
                  education_place = speaker_row['EduPlace'].values[0]
                  length_stay = speaker_row['Length stay in NL or VL'].values[0]
                  time_dutch_l2 = speaker_row['Time on Dutch L2'].values[0]
                  duration = recordings_row['Duration (seconds)'].values[0]

                  f_write.write(simplified_filename + ';' + \
                                speaker_id + ";" + \
                                component + ";" + \
                                str(group) + ";" + \
                                str(age) + ";" + \
                                str(gender) + ";" + \
                                str(cef) + ";" + \
                                str(dialect_region) + ";" + \
                                str(res_place) + ";" + \
                                str(birth_place) + ";" + \
                                str(home_language_1) + ";" + \
                                str(home_language_2) + ";" + \
                                str(comment) + ";" + \
                                str(education) + ";" + \
                                str(education_place) + ";" + \
                                str(length_stay) + ";" + \
                                str(time_dutch_l2) + ";" + \
                                str(duration) + ";" + \
                                str(wer_result) + ';' + \
                                str(cer_result) + ';' + \
                                reference_text + ";" + \
                                predicted_text + "\n")
                  f_write.flush()

                  f_write_log.write(simplified_filename + " " + speaker_id + "\n\n" + \
                          "reference:\n" + 
                          reference_text + "\n\n" + 
                          "prediction:\n" + 
                          predicted_text + "\n" + \
                          "wer: {} cer: {}\n\n".format(wer_result, cer_result))

                  f_write_log.flush()
                            
if __name__ == "__main__":
    main(sys.argv)
