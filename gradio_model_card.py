import gradio as gr
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from evaluate_model import get_error_rate

subgroup_1 = pd.DataFrame({'wer': [0.40, 0.41, 0.45, 0.42], 'cer': [0.20, 0.21, 0.45, 0.42], 'subgroup': [1, 1, 1, 1], 'gender': 'F'})
subgroup_2 = pd.DataFrame({'wer': [0.40, 0.41, 0.45, 0.42], 'cer': [0.30, 0.31, 0.45, 0.42], 'subgroup': [2, 2, 2, 2], 'gender': 'M'})

# subgroup_1 = get_error_rate(1)
# subgroup_2 = get_error_rate(2)

data = pd.concat([subgroup_1, subgroup_2], ignore_index=True)
data.head(10)
base = [[1, 35.3], [2, 18.4], [3,  55.1], [4, 56.9], [5, 24.2]]
baseline = pd.DataFrame(base, columns=['subgroup', 'baseline'])

def get_mean(df):
    df['wer'] = df['wer'].apply(lambda x: x*100)
    df.groupby('subgroup')['wer'].mean() 
    df = df.merge(baseline, how='left', on='subgroup')
    return df

def launch_gradio(df):
    wer = plt.figure(1, figsize=(10, 5))
    for ind in df.index:
        X_axis_wer = np.arange(len(df))
        print(X_axis_wer)
        plt.bar(X_axis_wer - 0.1, df['wer'][ind], 0.2, label = 'wer', color='blue')
        plt.bar(X_axis_wer + 0.1, df['baseline'][ind], 0.2, label = 'baseline', color='orange')
        plt.xticks(X_axis_wer)
    plt.xlabel("Group")
    plt.ylabel("Word error rate")
    plt.title("Whisper subgroup comparison", fontsize=14, fontweight='bold')
    plt.grid(False, 'both', 'x')
    plt.grid(True, 'both', 'y')
    plt.legend()

    cer = plt.figure(2, figsize=(10, 5))
    for ind in df.index:
        X_axis_cer = np.arange(len(df))
        # plt.bar(X_axis_cer, df['wer'][ind], 0.4, label = 'cer')
        plt.bar(X_axis_wer - 0.1, df['wer'][ind], 0.2, label = 'cer', color='blue')
        plt.bar(X_axis_cer + 0.1, df['baseline'][ind], 0.2, label = 'baseline', color='orange')
        plt.xticks(X_axis_cer, df['subgroup'])
    plt.xlabel("Group")
    plt.ylabel("Character error rate")
    plt.title("Whisper subgroup comparison", fontsize=14, fontweight='bold')
    plt.grid(False, 'both', 'x')
    plt.grid(True, 'both', 'y')
    plt.legend()

    boxplot = px.box(data, x='subgroup', y="wer", hover_data=['gender'])

    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Model card
        In this page we visualize the results of our whisper model (large-v2) 
        We measure the results by the Word Error Rate and the Character Error Rate
        These values are compared with a baseline. 
        """
        )
        gr.Markdown(
            """ 
            ## Boxplot-information
            The box plot below shows the distribution of the Word Error Rate (WER) for different subgroups. 

            This plot helps to gain insight into the performance of the model for different subgroups and to compare the results with the baseline.

            ## Baseline-information
            The baseline in this model card is derived from a study in which the average score of the baseline was established. This score serves as a reference point for evaluating the performance of our model. In the box plot, the baseline is represented as a blue line that indicates the average WER for each subgroup. By comparing the WER of the model with the baseline, we can understand how well our model performs compared to the established standard.

            ### References
            Quantifying Bias in Automatic Speech Recognition, Siyuan Feng, Olya Kudina, Bence Mark Halpern and Odette Scharenborg, link: https://drive.google.com/file/d/18Y60qr4SAX21-kdKiyg4QxbWaUtsJ4gR/view?usp=sharing
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("WER"):
                gr.Plot(wer)
            with gr.TabItem("CER"):
                gr.Plot(cer)
            with gr.TabItem("Boxplot"):
                gr.Plot(boxplot)

    if __name__ == "__main__":
        demo.launch()

launch_gradio(get_mean(data))
