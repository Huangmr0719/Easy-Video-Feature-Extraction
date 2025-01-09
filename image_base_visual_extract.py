from PIL import Image
import torch
from transformers import AutoModel, CLIPImageProcessor
from pathlib import Path
import json
from tqdm import tqdm
import os

def save_image_features(img_feats, name_ids, save_folder):
    """
    Save image features to a .pt file in a specified folder.

    Args:
    - img_feats (torch.Tensor): Tensor containing image features
    - name_ids (str): Identifier to include in the filename
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    torch.save(img_feats, filepath)

def clip_es(csv_path, save_folder, model_name_or_path="BAAI/EVA-CLIP-8B", image_size=224):
    """
    Process extracted video frames to compute image features using CLIP and save the features.

    Args:
    - csv_path (str): Path to the CSV file generated by `extract_es`.
    - save_folder (str): Directory to save extracted image features.
    - model_name_or_path (str): CLIP model path or name. Default is "BAAI/EVA-CLIP-8B".
    - image_size (int): Image size for the CLIP processor. Default is 224.

    Returns:
    - None
    """
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and processor
    model = AutoModel.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)

    # Read the CSV file
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Process each video folder
    pbar = tqdm(total=len(df), desc="Processing video frames")

    for _, row in df.iterrows():
        feature_path = Path(row["feature_path"])
        video_name = feature_path.name

        # Skip if features are already saved
        feature_save_path = os.path.join(save_folder, f"{video_name}.pt")
        if os.path.exists(feature_save_path):
            pbar.update(1)
            continue

        # Ensure the frame folder exists
        if not feature_path.exists():
            print(f"Frame folder not found: {feature_path}")
            pbar.update(1)
            continue

        # Collect and process all frames in the folder
        image_paths = sorted(
            feature_path.glob("*.jpg"), 
            key=lambda x: int(x.stem)  # Sort by frame number
        )
        img_feature_list = []

        for image_path in image_paths:
            image = Image.open(str(image_path)).convert("RGB")
            input_pixels = processor(images=image, return_tensors="pt", padding=True).pixel_values.to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(input_pixels)
                img_feature_list.append(image_features)

        # Stack all frame features into a single tensor
        img_feature_tensor = torch.stack(img_feature_list)
        img_feats = img_feature_tensor.squeeze(1)

        # Save features
        save_image_features(img_feats, video_name, save_folder)
        pbar.update(1)

    pbar.close()
    print("Feature extraction complete.")

if __name__ == '__main__':
    make_csv.make_path_csv(args.csv, args.extracted)
    # Path to the CSV file generated by `extract_es`
    csv_path = "/path/to/extracted_video_frames.csv"  # Replace with your actual path

    # Folder to save extracted features
    save_folder = "/path/to/feature_save_folder"  # Replace with your actual path

    # Run the CLIP-based feature extraction
    clip_es(csv_path, save_folder)