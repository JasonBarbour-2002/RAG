import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class Sentences:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
        )
        self.model.max_seq_length = 8192
        self.df = pd.read_csv("Processed/transcript_processed.csv")
        self.documents = self.df["text"].tolist()
        self.document_embeddings = None

    def compute_document_features(self, save_path=None, precompute_path=None):
        embedding_file = os.path.join(precompute_path, "document_embeddings.npy") if precompute_path else None

        if embedding_file and os.path.exists(embedding_file):
            self.document_embeddings = torch.from_numpy(np.load(embedding_file)).to(self.device)
            return self.document_embeddings, self.documents

        with torch.no_grad():
            self.document_embeddings = self.model.encode(
                self.documents,
                device=self.device.type,
                convert_to_tensor=True
            )

        if save_path:
            np.save(os.path.join(save_path, "document_embeddings.npy"), self.document_embeddings.cpu().numpy())

        return self.document_embeddings, self.documents

    def compute_query_features(self, prompts):
        query_embeddings = self.model.encode(
            prompts,
            prompt_name="query",
            device=self.device.type,
            convert_to_tensor=True
        )
        return query_embeddings

    def get_top_k_documents(self, query, similarity_fn, save_path=None, precompute_path=None, threshold=None, **kwargs):
        document_embeddings, documents = self.compute_document_features(save_path=save_path, precompute_path=precompute_path)
        query_embeddings = self.compute_query_features(query)
        query_embeddings = query_embeddings.reshape(1, -1)
        
        topk_ind, topk_sim = similarity_fn(query_embeddings, document_embeddings, threshold=threshold, **kwargs)

        if len(topk_ind.shape) == 1:
            if topk_ind.shape[0] == 0:
                return topk_ind, topk_sim

        topk_times = self.df.iloc[topk_ind]["start"]
        print(f"topk_times: {topk_times}")
        if isinstance(topk_times, pd.Series):
            topk_times = topk_times.to_numpy()
        elif isinstance(topk_times, str):
            topk_times = np.array([topk_times])  
            topk_sim = np.array([topk_sim])

        # Optional: adjust this if 'start' is a list/array or string you want to parse
        return topk_times, topk_sim
