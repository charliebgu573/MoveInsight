import pandas as pd
import numpy as np
import cv2
import subprocess

def detect_hits(csv_file, angle_threshold=100):
    df = pd.read_csv(csv_file)
    df = df[df['Visibility'] == 1].copy()
    df = df.sort_values(by='Frame').reset_index(drop=True)

    # Calculate displacement and angle
    df['dx'] = df['X'].diff()
    df['dy'] = df['Y'].diff()
    df['angle'] = np.degrees(np.arctan2(df['dy'], df['dx']))

    # Calculate angle change
    df['angle_change'] = df['angle'].diff().abs()
    df['angle_change'] = df['angle_change'].apply(lambda x: 360 - x if x > 180 else x)

    # Detect hits where the angle change is significant
    hit_frames = df[df['angle_change'] > angle_threshold]['Frame'].tolist()

    # Remove hits that are too close to each other (likely duplicates)
    if not hit_frames:
        return []

    unique_hits = [hit_frames[0]]
    for frame in hit_frames[1:]:
        if frame - unique_hits[-1] > 30:  # Assuming at least 1 second between hits
            unique_hits.append(frame)

    return unique_hits

def segment_video(video_path, hit_frames, output_prefix='hit_segment'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()

    for i, frame in enumerate(hit_frames):
        hit_time = frame / fps
        start_time = max(0, hit_time - 0.75)
        end_time = min(video_duration, hit_time + 0.75)
        duration = end_time - start_time

        if duration > 0:
            output_filename = f"{output_prefix}_{i+1}.mp4"
            command = [
                'ffmpeg',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c', 'copy',
                output_filename
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error creating segment {output_filename}: {e}")
                print(f"FFmpeg stderr: {e.stderr.decode()}")

if __name__ == "__main__":
    csv_file = 'test_shuttlecock_tracking.csv'
    video_file = 'test_shuttlecock_tracked.mp4'

    # Get video properties (already done in segment_video, but keeping for initial print)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}")
        cap.release()

        hit_frames = detect_hits(csv_file)
        print(f"Detected hit frames: {hit_frames}")

        if hit_frames:
            segment_video(video_file, hit_frames)
        else:
            print("No hits detected.")