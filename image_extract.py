import cv2
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
import os
import logging

import utils.make_path_csv as make_csv

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def extract_es(csv_path, framerate=1):
    """
    Reads video paths and feature paths from a CSV file, extracts video frames, and saves them as JPEG images.
    If the output folder already exists and contains files, skips processing for that video.
    
    Args:
        csv_path (str): Path to the CSV file, expected to contain 'video_path' and 'feature_path' columns.
        framerate (int): Number of frames to extract per second. Default is 1 frame/second.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV file from {csv_path}")
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found.")
        return
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return

    # Initialize progress bar
    pbar = tqdm(total=len(df), desc="Processing videos")

    for index, row in df.iterrows():
        video_path = row['video_path']
        feature_path = row['feature_path']

        # Create Path objects
        video_fp = Path(video_path)
        output_path = Path(feature_path)

        # Check if the output folder already exists and is non-empty
        if output_path.exists() and any(output_path.iterdir()):
            print(f"Output folder {output_path} already exists and contains files. Skipping video: {video_path}")
            pbar.update(1)
            continue

        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Open the video
        vidcap = cv2.VideoCapture(str(video_fp))
        if not vidcap.isOpened():
            print(f"Unable to open video: {video_path}")
            pbar.update(1)
            continue

        count = 0
        success = True
        fps = framerate  # Frames to extract per second
        fps_ori = vidcap.get(cv2.CAP_PROP_FPS)

        # Handle videos with a frame rate of 0
        if fps_ori == 0:
            print(f"Video frame rate is 0. Skipping video: {video_path}")
            vidcap.release()
            pbar.update(1)
            continue

        frame_interval = max(int(fps_ori / fps), 1)  # Ensure frame interval is at least 1

        while success:
            success, image = vidcap.read()
            if not success:
                break
            if count % frame_interval == 0:
                frame_number = count // frame_interval
                frame_filename = output_path / f"{frame_number}.jpg"
                # Save frame as a JPEG image
                cv2.imwrite(str(frame_filename), image)
            count += 1

        vidcap.release()
        pbar.update(1)

    pbar.close()
    print("All videos processed successfully.")

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Provide your CSV file path
    csv_path = '/your/CSV/file/path'
    make_csv.make_path_csv(csv_path, "images")

    extract_es(csv_path, framerate=1)  # Extract 1 frame per second