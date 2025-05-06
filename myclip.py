#%%
import os
import clip
import torch
import numpy as np
from PIL import Image

class CLIP:
    def __init__(self, model_name="ViT-B/32", device="mps"):
        self.device = self.get_device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_device(self, device):
        match device:
            case "mps":
                if torch.backends.mps.is_available():
                    return "mps"
                else:
                    print("MPS is not available on this device. Using CPU instead.")
                    return "cpu"
            case "cuda":
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    print("CUDA is not available on this device. Using CPU instead.")
                    return "cpu"
            case "cpu":
                return "cpu"
            case _:
                raise ValueError("Invalid device. Choose 'mps', 'cuda', or 'cpu'.")
            
    def compute_image_features(self, path="", save_path=None, precompute_path=None):           
        files = [f for f in os.listdir(path) if f.endswith('.jpg')]
        files = sorted(files)
        self.images = []
        for f in files:
            image = Image.open(os.path.join(path, f))
            self.images.append(image)
        
        if precompute_path and os.path.exists(precompute_path + "/image_features.pt"):
            self.image_features = torch.load(precompute_path + "/image_features.pt")
            return self.image_features, self.images
        on_GPU = torch.stack([self.preprocess(image) for image in self.images]).to(self.device)

        with torch.no_grad():
            self.image_features = self.model.encode_image(on_GPU)
            
            self.image_features /= self.image_features.norm(dim=-1, keepdim=True)
            # save image features
            if save_path:
                torch.save(self.image_features, save_path + "/image_features.pt")
        return self.image_features, self.images


    def compute_query_features(self, prompts):
        text = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text)
            
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        return self.text_features

    def get_image_times(self, query, similarity,  path="", save_path=None, precompute_path=None, threshold=None, **kwargs):
        video_info = np.loadtxt(path + "/video_info.csv", delimiter=",", skiprows=1)
        self.compute_image_features(path, save_path, precompute_path=precompute_path)
        self.compute_query_features(query)
        # Compute similarity
        topk_ind, topk_sim = similarity(self.text_features, self.image_features, threshold=threshold, **kwargs)
        # Check if the topk_ind is empty   
        if len(topk_ind.shape) == 1:
            if topk_ind.shape[0] == 0:
                return topk_ind, topk_sim
        # Get the topk image times
        topk_times = video_info[topk_ind]
        if len(topk_times.shape) == 1:
            topk_times = np.expand_dims(topk_times, axis=0)
        topk_times = topk_times[:, 3]
        return topk_times, topk_sim
        
        
# %%
