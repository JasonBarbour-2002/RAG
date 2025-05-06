import pandas as pd

class LexicalSearch:
    def __init__(self):
        self.df = pd.read_csv("Processed/transcript_processed.csv")
        self.documents = self.df["text"].tolist()

    def get_top_k_documents(self, query, similarity_fn, threshold=None, k=10):
        topk_ind, topk_sim = similarity_fn(query, self.documents, threshold=threshold, topk=k)
        
        if len(topk_ind) == 0:
            return topk_ind, topk_sim
        
        topk_times = self.df.iloc[topk_ind]["start"].to_numpy()
        return topk_times, topk_sim
        
        