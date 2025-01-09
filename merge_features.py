import numpy as np
import argparse
import os
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Feature merger")

parser.add_argument("--folder", type=str, required=True, help="folder of features")
parser.add_argument(
    "--output_path", type=str, required=True, help="output path for features"
)
parser.add_argument(
    "--pad",
    type=int,
    help="set as diff of 0 to trunc and pad up to a certain nb of seconds",
    default=0,
)

args = parser.parse_args()
files = os.listdir(args.folder)
files = [x for x in files if x[-4:] == ".npy"]

features = {}
for i in tqdm(range(len(files))):
    x = files[i]
    feat = torch.from_numpy(np.load(os.path.join(args.folder, x)))
    if args.pad and len(feat) < args.pad:
        feat = torch.cat([feat, torch.zeros(args.pad - len(feat), feat.shape[1])])
    elif args.pad:
        feat = feat[: args.pad]
    features[x] = feat.half()

torch.save(features, args.output_path)
