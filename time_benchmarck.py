#%%
import time
import numpy as np
from SQL import SQL
import matplotlib.pyplot as plt
from utils import cos_sim_numpy, faiss_similarity, ivfflat, hnsw
import faiss
#%%
np.random.seed(42)
embeddings = np.random.rand(100_000, 1000).astype(np.float32)
query = np.random.rand(1, 1000).astype(np.float32)
top_k = 1000
times = np.zeros((4))
#%%
# Init SQL 
sql = SQL()
sql.connect()
sql.add_vectors(embeddings.T)
# %%
# Init FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
#%% 
# Benchmarking cos_sim 
ts = time.process_time()
similar_indices_knn, similar_scores_knn = cos_sim_numpy(query, embeddings, topk=top_k)
te = time.process_time() - ts
times[0] = te
print(f"cos_sim took {te:.5f} seconds")
#%% 
# Benchmarking faiss_similarity
ts = time.process_time()
similar_indices_faiss, similar_scores_faiss = faiss_similarity(query, embeddings, topk=top_k, index=index)
te = time.process_time() - ts
times[1] = te
print(f"faiss_similarity took {te:.5f} seconds")
#%%
# Benchmarking ivfflat
ts = time.process_time()
similar_indices_ivfflat, similar_scores_ivfflat = ivfflat(query, embeddings, topk=top_k, sql=sql)
te = time.process_time() - ts
times[2] = te
print(f"ivfflat took {te:.5f} seconds")
#%%
# Benchmarking hnsw
ts = time.process_time()
similar_indices_hnsw, similar_scores_hnsw = hnsw(query, embeddings, topk=top_k, sql=sql)
te = time.process_time() - ts
times[3] = te
print(f"hnsw took {te:.5f} seconds")
#%%
plt.bar(['cos_sim', 'faiss_similarity', 'ivfflat', 'hnsw'], times)
plt.ylabel('Time (seconds)')
plt.title('Benchmarking Time for Different Similarity Functions')
plt.savefig('Results/benchmarking_time.png')
plt.show()
#%%