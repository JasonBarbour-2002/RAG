import atexit
import pandas as pd
import streamlit as st
from myclip import CLIP
from sentences import Sentences
from utils import cos_sim, faiss_similarity, bm25_similarity, tfidf_similarity, hnsw, ivfflat
from lexical import LexicalSearch
from SQL import SQL

st.set_page_config(
    page_title="RAG",
    page_icon=":guardsman:", 
    layout="centered"
)

NO_EMBEDDING_MODELS = ['BM25', 'TF-IDF']

# Init SQL 
sql = SQL()
sql.connect()

def cleanup():
    sql.close()
    
atexit.register(cleanup)

@st.cache_resource
def load_clip():
    return CLIP()
@st.cache_resource
def load_sentences():
    return Sentences()

#### Main Page ####
def to_time(seconds):
    print(f"seconds: {seconds}")
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    result = ""
    if hours > 0:
        result += f"{int(hours)}h "
    if minutes > 0:
        result += f"{int(minutes)}m "
    if seconds > 0:
        result += f"{int(seconds)}s "
    if result == "":
        result = "0s"
    return result

def parse_time(time_str):
    # replace first : by h 
    # second : by m
    # third : by s
    hours, minutes, seconds, _ = time_str.split(":")
    seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    return to_time(seconds)

def load_docs(path):
    df = pd.read_csv(path)
    docs = df["text"].tolist()
    return docs

def get_seach_algorithm():
    option = st.session_state.get("option", "KNN")
    match option:
        case "KNN":
            return cos_sim, {}
        case "FAISS":
            return faiss_similarity, {}
        case "IVFFLAT index":
           return ivfflat, {"sql": sql}
        case "HNSW index":
            return hnsw, {"sql": sql}
        case "TF-IDF":
            return tfidf_similarity, {}
        case "BM25":
            return bm25_similarity, {}
        case _:
            raise ValueError("Invalid algorithm selected.")

def get_threshold(algorithm):
    option = st.session_state.get("option", "KNN")
    if algorithm == "CLIP":
        dict_threshold = {
            "KNN": 0.25,
            "FAISS": 0.25,
            "IVFFLAT index": 0.25,
            "HNSW index": 0.25,
        }
        return dict_threshold.get(option, None)
    elif algorithm == "Sentences":
        dict_threshold = {
            "KNN": 0.55,
            "FAISS": 0.55,
            "IVFFLAT index": 0.55,
            "HNSW index": 0.55,
            
        }
        return dict_threshold.get(option, None)
    else:
        dict_threshold = {
            "TF-IDF": 0.3,
        }
        return dict_threshold.get(option, None)
                
def run_query():
    query = st.session_state.query
    sim_algo, kwargs = get_seach_algorithm()
    if st.session_state.get("option", "KNN") in NO_EMBEDDING_MODELS:
        lexical = LexicalSearch()
        topk_times_lexical, topk_sim = lexical.get_top_k_documents(query, sim_algo, threshold=get_threshold("Lexical"))
        if len(topk_times_lexical) == 0:
            st.session_state.time_sentence = -1
        else:
            st.session_state.time_sentence = parse_time(topk_times_lexical[0])
            st.session_state.sim_sentence = topk_sim[0]
        return
    my_clip = load_clip()
    sentence_model = load_sentences()
    Dir = "Processed/Slides"
    topk_times_clip, topk_sim =  my_clip.get_image_times(query, sim_algo, precompute_path=Dir, path=Dir, save_path=Dir, threshold=get_threshold("CLIP"), **kwargs)
    if len(topk_times_clip) == 0:
        st.session_state.time_clip = -1
    else:
        st.session_state.time_clip = to_time(topk_times_clip[0])
        st.session_state.sim_clip = topk_sim[0]
    Dir = "Processed"
    topk_times_sentence, topk_sim = sentence_model.get_top_k_documents(query, sim_algo, precompute_path=Dir, save_path=Dir, threshold=get_threshold("Sentences"), **kwargs)
    if len(topk_times_sentence) == 0:
        st.session_state.time_sentence = -1
    else:
        st.session_state.time_sentence = parse_time(topk_times_sentence[0])
        st.session_state.sim_sentence = topk_sim[0]
    
def update_sentence():
    st.session_state.update(start_time=st.session_state.time_sentence)

def update_clip():
    st.session_state.update(start_time=st.session_state.time_clip)
        
def main():
    # Title
    st.title("Video Retrieval System")
    st.subheader("Retrieval-Augmented Generation (RAG)")
    # Video 
    start_time = st.session_state.get("start_time", 0)
    st.video(
        "Data/video.mp4", 
        start_time=start_time, 
        subtitles={"English": "Processed/transcript.srt"},
    )
    st.text("Parameterized Complexity of token sliding, token jumping - Amer Mouawad")
    st.text_input("Enter your query:", on_change=run_query, key="query")
    st.session_state.debugg = st.session_state.get("debugg", '')
    st.text(st.session_state.debugg)

    # Display the results
    query = st.session_state.get("query", "") 
    if query == "":
        # No query entered
        return 
    else:
        st.subheader("Results")
    if st.session_state.get("option", "KNN") in NO_EMBEDDING_MODELS:
        time_sentence = st.session_state.get("time_sentence", None)
        print(f"time_sentence: {time_sentence}")
        if time_sentence is not None:
            if time_sentence == -1:
                st.info("No results found in the transcript.")
            else:
                st.info(f"Results found in the transcript at {time_sentence}.\t Similarity: {st.session_state.sim_sentence:.2f}")
                st.button("Go to time code", on_click=update_sentence, key="time_sentence_button")
    else:
        time_clip = st.session_state.get("time_clip", None)
        if time_clip is not None:
            if time_clip == -1:
                st.info("No results found in the slides.")
            else:
                st.info(f"Results found in the slides at {time_clip}.\t Similarity: {st.session_state.sim_clip:.2f}")
                st.button("Go to time code", on_click=update_clip, key="time_clip_button")
            
        time_sentence = st.session_state.get("time_sentence", None)
        if time_sentence is not None:
            if time_sentence == -1:
                st.info("No results found in the transcript.")
            else:
                st.info(f"Results found in the transcript at {time_sentence}.\t Similarity: {st.session_state.sim_sentence:.2f}")
                st.button("Go to time code", on_click=update_sentence, key="time_sentence_button")
        

def test_input(query):
    st.session_state.query = query
    run_query()
    
    

def sidebar():
    st.sidebar.header("Search Algorithm")
    algorithms = [
        "KNN",
        "FAISS",
        "IVFFLAT index", 
        "HNSW index",
        "TF-IDF",
        "BM25",
    ]
    st.sidebar.selectbox(
        "Select the algorithm to use:", 
        algorithms,
        key="option",
        on_change=run_query
    )
    
    st.sidebar.header("Gold Test Set")
    st.sidebar.text("Click on a button to test the query.")
    st.sidebar.subheader("Answerable questions:")
    query1 = "What is a reconfiguration graph?"
    st.sidebar.text("Prompt 1: \nExpected: 6m")
    st.sidebar.button(query1, key="test_set1", on_click=lambda: test_input(query1))
    
    query2 = "What is the time complexity of token sliding?"
    st.sidebar.text("Prompt 2: \nExpected: 23m")
    st.sidebar.button(query2, key="test_set2", on_click=lambda: test_input(query2))
    
    query3 = "what is the name of the professor being welcomed to the PC seminar?"
    st.sidebar.text("Prompt 3: \nExpected: 0s")
    st.sidebar.button(query3, key="test_set3", on_click=lambda: test_input(query3))
    
    query4 = "what is parametrized complexity?"
    st.sidebar.text("Prompt 4: \nExpected: 23m50s")
    st.sidebar.button(query4, key="test_set4", on_click=lambda: test_input(query4))
    
    query5 = "What is the PSPACE?"
    st.sidebar.text("Prompt 5: \nExpected: 17m50s")
    st.sidebar.button(query5, key="test_set5", on_click=lambda: test_input(query5))
    
    query6 = "What is the k-SAT problem?"
    st.sidebar.text("Prompt 6: \nExpected: 9m48s")
    st.sidebar.button(query6, key="test_set6", on_click=lambda: test_input(query6))
    
    query7 = "For what values of k is the graph coloring problem NP-complete?"
    st.sidebar.text("Prompt 7: \nExpected: 11m28s")
    st.sidebar.button(query7, key="test_set7", on_click=lambda: test_input(query7))
    
    query8 = "What is token jumping?"
    st.sidebar.text("Prompt 8: \nExpected: 13m15s")
    st.sidebar.button(query8, key="test_set8", on_click=lambda: test_input(query8))
    
    query9 = "The complexity of reconfiguration in a table"
    st.sidebar.text("Prompt 9: \nExpected: 21m56s")
    st.sidebar.button(query9, key="test_set9", on_click=lambda: test_input(query9))
    
    query10 = "What is the buffer technique?"
    st.sidebar.text("Prompt 10: \nExpected: 42m03s")
    st.sidebar.button(query10, key="test_set10", on_click=lambda: test_input(query10))
    
    
    st.sidebar.subheader("Unanswerable questions:")
    
    
    query11 = "What is a network embedding?"
    st.sidebar.text("Prompt 11: \nExpected: Not found")
    st.sidebar.button(query11, key="test_set11", on_click=lambda: test_input(query11))

    query12 = "What is FPT?"
    st.sidebar.text("Prompt 12: \nExpected: Not found")
    st.sidebar.button(query12, key="test_set12", on_click=lambda: test_input(query12))
    
    query13= "What is a free graph?"
    st.sidebar.text("Prompt 13: \nExpected: Not found")
    st.sidebar.button(query13, key="test_set13", on_click=lambda: test_input(query13))
    
    query14 = "What is a Token?"
    st.sidebar.text("Prompt 14: \nExpected: Not found")
    st.sidebar.button(query14, key="test_set14", on_click=lambda: test_input(query14))
    
    query15 = "How old is the professor?"
    st.sidebar.text("Prompt 15: \nExpected: Not found")
    st.sidebar.button(query15, key="test_set15", on_click=lambda: test_input(query15))
    
    
    
if __name__ == "__main__":
    sidebar()
    main()
    