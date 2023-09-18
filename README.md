# RTL

## Welcome to the project RTL repo!

The goal of this project is to create a speech recognition model that outperforms the previous one created with Wav2vec2 on the Jasmin dataset. The Jasmin dataset is a spoken Dutch corpus containing children, elderly and non-native Dutch speakers. The model we are creating uses OpenAIs Whisper as the base model and will be finetuned to get lower error rates when transcribing the Jasmin dataset.

## Jasmin Dataset
The Jasmin dataset can be found on the HvA server. The data exists out of Dutch and Flemish spoken by children, elderly and non-native speakers. There are also two categories: comp-p for dialogues between a person and a computer and comp-q for text that is read out loud.

The dataset exists out of the following subgroups:
* Group 1: Native children aged 7-11 (DC)
* Group 2: Native children aged 12-16 (DT)
* Group 3: Non-native children (NNC)
* Group 4: Non-native adults (NNA)
* Group 5: Native adults above 65 (DOA)

## Model Card
To visualise the results of the model, we have created a model card. This model card can be found in the `gradio.ipynb` notebook. To be able to run this notebook, you need to have the following packages installed:
* `pip install gradio`
* `pip install matplotlib`
* `pip install pandas`
* `pip install numpy`
* `pip install datasets`