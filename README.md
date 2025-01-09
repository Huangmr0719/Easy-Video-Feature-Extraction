# EZ-Video-Feature-Extraction
A Simple Video Visual Feature Extraction Code Implementation in PyTorch

## File Description
`image_extract.py`: Extract video frames as JPG images at a frame rate of fps=1. FPS can be adjusted.

`image_base_visual_extract.py`: Extract feature files using the model based on the extracted frame images. The default model used is CLIP.

`visual_extract.py`: Directly extract video features into `.npy` files.

`merge_features.py`: Combine all the `.npy` files containing videos in the dataset into a single `.pth` feature file.

## Usage
Go to `/utils/make_path_csv.py` and modify your video path, dataset division, feature paths. Try to run and see if the correct CSV file is generated. This file can be called in any Python script for feature extraction.
