#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
positive = pd.read_csv("Results/positive.csv")
negative = pd.read_csv("Results/negative.csv")
positive.columns = positive.columns.str.strip()
negative.columns = negative.columns.str.strip()
positive.set_index("query", inplace=True)
negative.set_index("query", inplace=True)
#%%
pos_clip = positive[[c for c in positive.columns if c.endswith("-c")]]
pos_sent = positive[[c for c in positive.columns if not c.endswith("-c")]]
neg_clip = negative[[c for c in negative.columns if c.endswith("-c")]]
neg_sent = negative[[c for c in negative.columns if not c.endswith("-c")]]
#%%
pos_clip = pos_clip.rename(columns=lambda x: x.replace("-c", ""))
neg_clip = neg_clip.rename(columns=lambda x: x.replace("-c", ""))
pos_sent = pos_sent.rename(columns=lambda x: x.replace("-s", ""))
neg_sent = neg_sent.rename(columns=lambda x: x.replace("-s", ""))
#%%
pos_clip_sum = pos_clip.sum(axis=0)
neg_clip_sum = neg_clip.sum(axis=0)
pos_sent_sum = pos_sent.sum(axis=0)
neg_sent_sum = neg_sent.sum(axis=0)
#%%
# bar plot
plt.bar(
    pos_clip_sum.index,
    pos_clip_sum.values,
    label="Positive Clip",
    color="blue"
)
plt.ylabel("True Positive")
plt.title("Positive Clip")
plt.savefig("Results/positive_clip.png")
plt.show()
#%%
plt.bar(
    neg_clip_sum.index,
    neg_clip_sum.values,
    label="Negative Clip",
    color="red",
)
plt.ylabel("True Negative")
plt.title("Negative Clip")
plt.savefig("Results/negative_clip.png")
plt.show()
#%%
plt.bar(
    pos_sent_sum.index,
    pos_sent_sum.values,
    label="Positive Sent",
    color="green"
)
plt.ylabel("True Positive")
plt.title("Positive Sentences")
plt.savefig("Results/positive_sent.png")
plt.show()
#%%
plt.bar(
    neg_sent_sum.index,
    neg_sent_sum.values,
    label="Negative Sent",
    color="orange",
)
plt.ylabel("True Negative")
plt.title("Negative Sentences")
plt.savefig("Results/negative_sent.png")
plt.show()
# %%
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].bar(
    pos_clip_sum.index,
    pos_clip_sum.values,
    label="Positive Clip",
    color="blue"
)
ax[0, 0].set_ylabel("True Positive")
ax[0, 0].set_title("Positive Clip")
ax[0, 0].set_ylim(0, 10)

ax[0, 1].bar(
    neg_clip_sum.index,
    neg_clip_sum.values,
    label="Negative Clip",
    color="red",
)
ax[0, 1].set_ylabel("True Negative")
ax[0, 1].set_title("Negative Clip")
ax[0, 1].set_ylim(0, 5)

ax[1, 0].bar(
    pos_sent_sum.index,
    pos_sent_sum.values,
    label="Positive Sent",
    color="green"
)
ax[1, 0].set_ylabel("True Positive")
ax[1, 0].set_title("Positive Sentences")
ax[1, 0].set_ylim(0, 10)

ax[1, 1].bar(
    neg_sent_sum.index,
    neg_sent_sum.values,
    label="Negative Sent",
    color="orange",
)
ax[1, 1].set_ylabel("True Negative")
ax[1, 1].set_title("Negative Sentences")
ax[1, 1].set_ylim(0, 5)
plt.tight_layout()
plt.savefig("Results/positive_negative.png")
plt.show()