#%%
import pandas as pd
#%%

def process_transcript(path="RAG/Processed/transcript.csv",window_size=5, stride=3):
    df = pd.read_csv(path, index_col=0)
    df_output = pd.DataFrame(columns=["index","start", "end", "text"])
    indeces = []
    start_times = []
    end_times = []
    texts = []
    for index, i in enumerate(range(0, len(df) - window_size + 1, stride)):
        start_time = df.iloc[i]["start"]
        end_time = df.iloc[i + window_size - 1]["end"]
        text = " ".join(df.iloc[i:i + window_size]["text"].tolist())
        indeces.append(index)
        start_times.append(start_time)
        end_times.append(end_time)
        texts.append(text)

    df_output["index"] = indeces
    df_output["start"] = start_times
    df_output["end"] = end_times
    df_output["text"] = texts
    return df_output

if __name__ == "__main__":
    df = process_transcript()
    df.to_csv("RAG/Processed/transcript_processed.csv", index=False)
    print(df.head())