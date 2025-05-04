# 🎓 Homework 5: Retrieval-Augmented Generation (RAG) for Video Question Answering

This repository contains the code for **Homework 5** of the course **"LLMs and RAGs"** at the **American University of Beirut**. The objective is to implement a **Retrieval-Augmented Generation (RAG)** model for **video question answering**.

---

## 📦 Requirements

It is recommended to use `conda` to create a virtual environment for this homework:

```bash
conda create -n rag python=3.11
```

Activate the environment:

```bash
conda activate rag
```
Finally, install the required packages:

```bash
conda env create -f environment.yml
```

---

## 🎬 Preprocessing the Data

The data used is based on the YouTube video:  
**[Parameterized Complexity of Token Sliding, Token Jumping – Amer Mouawad](https://research.google.com/youtube8m/)**

The preprocessed data is already available in the `Processed` folder.

If you'd like to use a different video, follow these steps:

### 1️⃣ Download the Video

Use the `download_youtube.py` script. Replace the `video_url` variable in the script with the desired URL. The video will be saved in the `Data` folder.

### 2️⃣ Run Preprocessing

Use the `preprocess.py` script. It performs the following steps:

- 🎧 **Audio Extraction**: Extracts audio from the video  
- 🗣️ **Transcription**: Transcribes audio using [Whisper](https://openai.com/research/whisper)  
- 📑 **Chunking**: Splits the transcription into text chunks  
- 🎞️ **Scene Detection**: Extracts scenes from the video

---
## 🛢️ Set Up Database Client

Some retrieval algorithms require a database to store embeddings.

Please install the appropriate version of **PostgreSQL** for your operating system:

- [PostgreSQL Downloads](https://www.postgresql.org/download/)

After installation, make sure the PostgreSQL server is running, and configure the connection credentials in your code or environment (e.g., host, port, username, password, and database name).

---
## 🚀 Run the App

Launch the Streamlit app with the following command:

```bash
streamlit run RAG.py
```
---

## 🧪 Features

- **Ask custom questions**: Enter any question related to the video, and the model will provide an answer based on the video content.
- **Choose retrieval algorithms**: Select from different algorithms to use for retrieving relevant information from the video.
- **Test with sample questions**: The app includes 15 sample questions:
  - ✅ **10** questions that are answerable from the video content.
  - ❌ **5** questions that are not answerable from the video content.

These features help you explore how well the RAG model performs in video question answering!
