import json
import csv
from os.path import exists

def make_path_csv(output_path, feature_name):
    """
    Generate a CSV file containing video paths, feature paths, and split labels for the ViTT dataset.

    Args:
        output_path (str): Path to save the generated CSV file.
        feature_name (str): Feature directory name for saving extracted features.

    Notes:
        - This script processes training, validation, and test split IDs for the ViTT dataset.
        - It generates video paths and feature paths based on dataset structure and includes a 'split' label for each record.
        - It checks if the required mapping files exist before processing.

    To adapt this script for another dataset:
        - Update the paths to split mapping files.
        - Modify the base directory structure and file format logic as needed.
    """
    # Define paths to the split mapping files
    train_ids_path = "/path/to/dataset/train_id_mapping.json"  # Replace with the path to your training split mapping file
    val_ids_path = "/path/to/dataset/val_id_mapping.json"      # Replace with the path to your validation split mapping file
    test_ids_path = "/path/to/dataset/test_id_mapping.json"    # Replace with the path to your test split mapping file
    
    # Check if mapping files exist
    if not exists(train_ids_path):
        print(f"Error: {train_ids_path} file does not exist.")
        return
    if not exists(val_ids_path):
        print(f"Error: {val_ids_path} file does not exist.")
        return
    if not exists(test_ids_path):
        print(f"Error: {test_ids_path} file does not exist.")
        return

    # Load training, validation, and test ID mappings
    with open(train_ids_path, 'r') as train_file:
        train_video_ids = json.load(train_file)
    with open(val_ids_path, 'r') as val_file:
        val_video_ids = json.load(val_file)
    with open(test_ids_path, 'r') as test_file:
        test_video_ids = json.load(test_file)
    
    # Initialize CSV data with header
    data = [["video_path", "feature_path", "split"]]

    cnt = 0  # Total count of videos
    train_cnt, val_cnt, test_cnt = 0, 0, 0

    # Process training split
    for short_id, video_id in train_video_ids.items():
        cnt += 1
        train_cnt += 1
        video_path = f"/path/to/videos/{video_id}.mp4"  # Replace with your video directory path
        feature_path = f"/path/to/features/{feature_name}/{video_id}"  # Replace with your feature directory path
        split = "train"
        data.append([video_path, feature_path, split])

    print(f"Training videos: {train_cnt}, Total records: {len(data)}")
    
    # Process validation split
    for short_id, video_id in val_video_ids.items():
        cnt += 1
        val_cnt += 1
        video_path = f"/path/to/videos/{video_id}.mp4"  # Replace with your video directory path
        feature_path = f"/path/to/features/{feature_name}/{video_id}"  # Replace with your feature directory path
        split = "val"
        data.append([video_path, feature_path, split])
    
    print(f"Validation videos: {val_cnt}, Total records: {len(data)}")

    # Process test split
    for short_id, video_id in test_video_ids.items():
        cnt += 1
        test_cnt += 1
        video_path = f"/path/to/videos/{video_id}.mp4"  # Replace with your video directory path
        feature_path = f"/path/to/features/{feature_name}/{video_id}"  # Replace with your feature directory path
        split = "test"
        data.append([video_path, feature_path, split])
    
    print(f"Test videos: {test_cnt}, Total records: {len(data)}")
    
    print(f"Total videos: {cnt}, Total records: {len(data)}")
    
    # Save data to CSV file
    try:
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerows(data)
        print(f"CSV saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving CSV to {output_path}: {e}")

if __name__ == '__main__':
    # Define the output CSV path and feature name
    output_csv_path = "/path/to/output/dataset.csv"  # Replace with the path to save your CSV file
    feature_name = "feature_name_example"  # Replace with your desired feature name

    # Generate the CSV file
    make_path_csv(output_csv_path, feature_name)