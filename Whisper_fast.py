import os
from faster_whisper import WhisperModel
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02}:{millis:03}"

def format_timestamp_srt(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02},{millis:03}"

def transcribe_video(path, results_path="Processed"):
    model = WhisperModel("medium.en", device="auto")
    # Transcribe
    segments, info = model.transcribe(path)
    with open(os.path.join(results_path, "transcript.srt"), "w") as srt:
        with open(os.path.join(results_path, "transcript.csv"), "w") as csv:
            csv.write("index,start,end,text\n")
            for i, segment in enumerate(segments, start=1):
                srt.write(f"{i}\n")
                csv.write(f"{i},{format_timestamp(segment.start)},{format_timestamp(segment.end)},\"{segment.text}\"\n")
                srt.write(f"{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}\n")
                print(f"[{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}] {segment.text}")
                srt.write(f"{segment.text}\n\n")

