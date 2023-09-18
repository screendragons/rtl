# Barcharts plotten en plaatsen in de gradio interface
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

def get_wer_chart(results, finetuned):
    index = [1, 2, 3, 4, 5]

    # Set the figure size
    wer = plt.figure(figsize=(10, 5))

    # Set the same y-axis scale for all charts
    #plt.ylim(0, 0.7)
    
    # Set the width of each bar
    bar_width = 0.2

    # Position of the bars on the x-axis
    r1 = np.arange(len(index))

    r2 = r1 + bar_width
    r3 = r2 + bar_width
    r4 = r3 + bar_width

    # Set the bar charts
    plt.bar(r1, finetuned['wer'], color='orange', width=bar_width)
    plt.bar(r2, results['wer'], color='blue', width=bar_width)
    plt.bar(r3, results['wav2vec2'], color='grey', width=bar_width)
    plt.bar(r4, results['baseline'], color='green', width=bar_width)

    # Set the labels
    plt.xlabel('Subgroup')
    plt.ylabel('WER')
    plt.title('WER per subgroup')

    # Set the x-axis ticks
    plt.xticks(r2 + bar_width, ['DC', 'DT', 'NNC', 'NNA', 'DOA'])

    # Set the legend
    plt.legend(['Finetuned model', 'Not finetuned model', 'wav2vec2', 'Baseline'])

    return wer

    

def get_cer_chart(results, finetuned):
    index = [1, 2, 3, 4, 5]

    # Set the figure size
    cer = plt.figure(figsize=(10, 5))

    # Set the same y-axis scale for all charts
    #plt.ylim(0, 0.7)

    # Set the width of each bar
    bar_width = 0.3

    # Position of the bars on the x-axis
    r1 = np.arange(len(index))
    r2 = r1 + bar_width

    # Set value on top of the bar
    #for index, value in enumerate(results['cer']):
        #plt.text(index - 0.12, value + 0.01, str(round(value, 3)))

    # Set the bar chart
    plt.bar(r1, finetuned['cer'], color='orange', width=0.3)
    plt.bar(r2, results['cer'], color='blue', width=0.3)

    # Set the labels
    plt.xlabel('Subgroup')
    plt.ylabel('CER')
    plt.title('CER per subgroup')

    # Set the x-axis ticks
    plt.xticks(r1 + bar_width, ['DC', 'DT', 'NNC', 'NNA', 'DOA'])

    # Set the legend
    plt.legend(['Whisper small finetuned', 'Whisper small v1'])

    return cer


with gr.Blocks() as demo:
    import pandas as pd

    # onfintuned model
    data = pd.read_csv("./files/statistics/eval-whisper-small-results-v1.txt", sep=';', header=0)

    # SUBGROUPS:
    # GROUP 1: native children aged 7-11 (DC)
    # GROUP 2: native children aged 12-16 (DT)
    # GROUP 3: non-native children (NNC)
    # GROUP 4: non-native adults (NNA)
    # GROUP 5: native adults above 65 (DOA)

    baseline_read = [[1, 0.353], [2, 0.184], [3,  0.551], [4, 0.569], [5, 0.242]]
    baseline_hmi = [[1, 0.434], [2, 0.353], [3, 0.616], [4, 0.613], [5, 0.395]]

    # Set wave2vec2 results per subgroup
    wav2vec2_read = [[1, 0.188], [2, 0.120], [3, 0.303], [4, 0.332], [5, 0.123]]
    wav2vec2_hmi = [[1, 0.312], [2, 0.250], [3, 0.475], [4, 0.501], [5, 0.307]]
    
    # Create dataframes for the bar charts
    wav2vec2_r = pd.DataFrame(wav2vec2_read, columns=['group', 'wav2vec2'])
    wav2vec2_h = pd.DataFrame(wav2vec2_hmi, columns=['group', 'wav2vec2'])
    
    baseline_r = pd.DataFrame(baseline_read, columns=['group', 'baseline'])
    baseline_h = pd.DataFrame(baseline_hmi, columns=['group', 'baseline'])

    # Create dataframes for the components from the onfinetuned model
    # comp-p = structured interviews = HMI (Human-machine interaction)
    # comp-q = read speech = READ
    
    comp_p = data[data['component'] == 'comp-p']
    comp_q = data[data['component'] == 'comp-q']

    comp_p = comp_p.groupby(['group'])[['wer', 'cer']].mean()
    comp_q = comp_q.groupby(['group'])[['wer', 'cer']].mean()

    # merge = updates the content of two DataFrame by merging them together
    comp_p = comp_p.merge(baseline_h, how='left', on=['group'])
    comp_q = comp_q.merge(baseline_r, how='left', on=['group'])

    comp_p = comp_p.merge(wav2vec2_h, how='left', on=['group'])
    comp_q = comp_q.merge(wav2vec2_r, how='left', on=['group'])

    # Finetuned model
    finetuned = pd.read_csv("./files/statistics/eval-rtl-whisper-small-results-v3.txt", sep=';', header=0)

    # Create dataframes for the components
    comp_p_finetuned = finetuned[finetuned['component'] == 'comp-p']
    comp_q_finetuned = finetuned[finetuned['component'] == 'comp-q']

    comp_p_finetuned = comp_p_finetuned.groupby(['group'])[['wer', 'cer']].mean()
    comp_q_finetuned = comp_q_finetuned.groupby(['group'])[['wer', 'cer']].mean()

    # Create graphs for the gradio interface
    wer_read = get_wer_chart(comp_q, comp_q_finetuned)
    cer_read = get_cer_chart(comp_q, comp_q_finetuned)
    wer_hmi = get_wer_chart(comp_p, comp_p_finetuned)
    cer_hmi = get_cer_chart(comp_p, comp_p_finetuned)

    gr.Markdown(
    """
    # Model card
    In this page we visualize the results of our whisper model (small-v1) 
    We measure the results by the Word Error Rate and the Character Error Rate
    These values are compared with a baseline. 
    """
    )
    gr.Markdown(
        """ 
        ### Subgroups   
        The subgroups are based on the age and language proficiency of the speaker. The subgroups are as follows:

        - DC: native children aged 7-11
        - DT: native children aged 12-16
        - NNC: non-native children
        - NNA: non-native adults
        - DOA: native adults above 65

        ### Word Error Rate (WER)
        The Word Error Rate (WER) is a metric that measures the performance of a speech recognition system. It is calculated by comparing the number of words that are incorrectly predicted by the model with the total number of words in the reference text.

        ### Character Error Rate (CER)
        The Character Error Rate (CER) is also metric that measures the performance of a speech recognition system. It is calculated by comparing the number of characters that are incorrectly predicted by the model with the total number of characters in the reference text.
         
        ## Baseline-information
        The baseline in this model card is derived from a study in which the average score of the baseline was established. This score serves as a reference point for evaluating the performance of our model. In the box plot, the baseline is represented as a blue line that indicates the average WER for each subgroup. By comparing the WER of the model with the baseline, we can understand how well our model performs compared to the established standard.

        ### References
        Quantifying Bias in Automatic Speech Recognition, Siyuan Feng, Olya Kudina, Bence Mark Halpern and Odette Scharenborg, link: https://drive.google.com/file/d/18Y60qr4SAX21-kdKiyg4QxbWaUtsJ4gR/view?usp=sharing
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Read"):
            gr.Plot(wer_read)
            gr.Plot(cer_read)

        with gr.TabItem("HMI"):
            gr.Plot(wer_hmi)
            gr.Plot(cer_hmi)

    if __name__ == "__main__":
        demo.launch()