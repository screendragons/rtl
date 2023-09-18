# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluate_model import get_error_rate
import subprocess
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
import constants





subgroup_1 = pd.DataFrame({'wer': [40, 41, 45, 42], 'cer': [20, 21, 45, 42], 'subgroup': [1, 1, 1, 1], 'gender': 'F'})
subgroup_2 = pd.DataFrame({'wer': [40, 41, 45, 42], 'cer': [30, 31, 45, 42], 'subgroup': [2, 2, 2, 2], 'gender': 'M'})

# subgroup_1 = get_error_rate(constants.DC_SUBGROUP_NR)
# subgroup_2 = get_error_rate(constants.DT_SUBGROUP_NR)
data = pd.concat([subgroup_1, subgroup_2], ignore_index=True)
data['wer'] = data['wer'].apply(lambda x: x*100)

base = [[1, 35.3], [2, 18.4], [3,  55.1], [4, 56.9], [5, 24.2]]
baseline = pd.DataFrame(base, columns=['subgroup', 'baseline'])

# Creating plot
fig = px.box(data, x='subgroup', y="wer", hover_data=['gender'], points="all")
# Added custom tick labels with "comp q" appended
fig.update_xaxes(tickvals=[1, 2], ticktext=[f"{label} comp q" for label in [1, 2]])
# fig.add_trace(go.Scatter(x=[1,2], y=[35.3, 18.4], mode='lines'))
# show plot
# fig.show()


demo = gr.Blocks()

with demo:
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
        with gr.Tab("Boxplot"):
            gr.Plot(fig)

# Launch the interface
demo.launch()