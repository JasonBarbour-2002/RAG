import os
from Whisper_fast import transcribe_video
from Process_transcript import process_transcript
from scene_detect import split_video_into_scenes

video = "Data/video.mp4"
results_path = "Processed"

#### Retrieve the video  transcript #####
os.makedirs(results_path, exist_ok=True)
# transcribe the video to get the transcript
transcribe_video(video, results_path=results_path)
# process the transcript to get the data batches
df = process_transcript(path=os.path.join(results_path, "transcript.csv"), window_size=5, stride=3)
df.to_csv(os.path.join(results_path, "transcript_processed.csv"), index=False)

#### Retrieve the video scenes #####
os.makedirs(os.path.join(results_path, "Slides"), exist_ok=True)
split_video_into_scenes(video, threshold=5.0, output_dir=os.path.join(results_path, "Slides"))
