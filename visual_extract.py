#modified from 

import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from utils.video_loader import VideoLoader
from torch.utils.data import DataLoader
from utils.preprocessing import Preprocessing
from utils.random_sequence_shuffler import RandomSequenceSampler
import utils.make_path_csv as make_csv

import clip

import ssl
ssl._create_default_https_context =ssl._create_unverified_context


parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--csv",
    type=str,
    help="input csv with columns video_path (input video) and feature_path (output path to feature)",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=1,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument(
    "--feature_dim", type=int, default=768, help="output video feature dimension"
)
parser.add_argument(
    "--model_path", type=str, default=None, help="clip path"
)
parser.add_argument(
    "--model_name", type=str, default='ViT-L/14', help="model name"
)
parser.add_argument(
    "--num_frames", type=int, default=16, help="the number of frames"
)
parser.add_argument(
    "--use_checkpoints", type=bool, default=False, help="whether or not to use checkpoints"
)
##
parser.add_argument(
    "--extracted", type=str, default="clipvitl14"
)
parser.add_argument(
    "--dataset_name", type=str, default="vitt"
)
args = parser.parse_args()

make_csv.make_path_csv(args.csv, args.extracted)

dataset = VideoLoader(
    args.csv,
    framerate=1,  # one feature per second max
    size=224,
    centercrop=True,
)
n_dataset = len(dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    sampler=sampler if n_dataset > 10 else None,
)
preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14", download_root=args.model_path)
model.eval()
model = model.cuda()

with th.no_grad():
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        if len(data["video"].shape) > 3: # [1, 132, 3, 224, 224]
            print(
                "Computing features of video {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            #print("data video0 shape::",data["video"].shape)
            video = data["video"].squeeze() # [132, 3, 224, 224]
            #print("data video1 shape::",video.shape)

            if len(video.shape) == 4:
                video = preprocess(video) # [132, 3, 224, 224]
                #print("data video2 shape::",video.shape)
                n_chunk = len(video) # 132
                #print(n_chunk)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                frame = 0

                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()

                    batch_features = model.encode_image(video_batch)# [1, 768]

                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    
                    features[min_ind:max_ind] = batch_features
                    frame += 1
                print("features shape::",features.shape) # [432, 768]
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")
                np.save(output_file, features)
        else:
            print("Video {} already processed.".format(input_file))