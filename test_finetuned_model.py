from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, pipeline
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import gradio as gr
import torch

model = WhisperForConditionalGeneration.from_pretrained("hannatoenbreker/whisper-dutch")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Dutch", task="transcribe")
pipe = pipeline(model="openai/whisper-small")
def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Dutch",
    description="Realtime demo for Dutch speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()