import os
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

def split_video_into_scenes(video_path, threshold=5.0, output_dir=''):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    # split_video_ffmpeg(video_path, scene_list, show_progress=True)
    
    # Save frames to output directory
    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    with open(os.path.join(output_dir, "video_info.csv"), 'w') as f:
        f.write("Scene_index, Start_frame, End_frame, Time_start, Time_end")
        for i, (start, end) in enumerate(scene_list):
            start_frame = start.get_frames()
            end_frame = end.get_frames()
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            f.write(f"\n{i:04d}, {start_frame}, {end_frame}, {start_time:.2f}, {end_time:.2f}")
            # Save last frame of the scene
            video.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
            ret, frame = video.read()
            if ret:
                save_path = os.path.join(output_dir, f'scene_{i:04d}.jpg')
                cv2.imwrite(save_path, frame)
                print(f"Saved scene {i:04d} at frame {end_frame}, time {start_time:.2f} -> {end_time:.2f}s")
            else:
                print(f"Failed to read frame {end_frame} for scene {i:04d}")
    video.release()
    print(f"Scenes saved to {output_dir}")
    return scene_list
# Example usage

if __name__ == "__main__":
    video_path = "RAG/data/video.mp4"
    output_dir = "RAG/Processed/Slides"
    split_video_into_scenes(video_path, threshold=5.0, output_dir=output_dir)