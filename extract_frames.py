#%%
import os
import cv2
import imagehash
from PIL import Image

video_path = 'RAG/Data/video.mp4'
output_dir = 'RAG/Processed/Slides'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps

frame_idx = 0
slide_idx = 0

last_saved_hash = None
buffered_frame = None
similarity_threshold = 1 
start_frame = 0

def compute_hash(frame):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.dhash(pil_img)

with open(output_dir + '/video_info.csv', 'w') as f:
    f.write(f"Slide_index, Start_frame, End_frame, Time_start, Time_end\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            # At end of video, save whatever frame was buffered
            if buffered_frame is not None:
                save_path = os.path.join(output_dir, f'slide_{slide_idx:04d}.jpg')
                cv2.imwrite(save_path, buffered_frame)
                print(f"Saved final slide {slide_idx:04d} at frame {frame_idx}, time {start_frame * frame_time:.2f} -> {frame_idx * frame_time:.2f}s")
                start_frame = frame_idx
                # Save info about frame in the CSV
                f.write(f"{slide_idx:04d}, {start_frame}, {frame_idx}, {start_frame * frame_time:.2f}, {frame_idx * frame_time:.2f}\n")
                
            break

        current_hash = compute_hash(frame)

        if last_saved_hash is None:
            # First frame ever
            buffered_frame = frame.copy()
            last_saved_hash = current_hash

        elif (current_hash - last_saved_hash) > similarity_threshold:
            # Significant change detected — save the last buffered frame
            save_path = os.path.join(output_dir, f'slide_{slide_idx:04d}.jpg')
            cv2.imwrite(save_path, buffered_frame)
            print(f"Saved slide {slide_idx:04d} at frame {frame_idx}, time {start_frame * frame_time:.2f} -> {frame_idx * frame_time:.2f}s")
            # Save info about frame in the CSV
            f.write(f"{slide_idx:04d}, {start_frame}, {frame_idx}, {start_frame * frame_time:.2f}, {frame_idx * frame_time:.2f}\n")
            slide_idx += 1
            start_frame = frame_idx + 1

            # Update for next
            buffered_frame = frame.copy()
            last_saved_hash = current_hash

        else:
            # Still similar — just update the buffer with fresher frame
            buffered_frame = frame.copy()

        frame_idx += 1

cap.release()
#%%