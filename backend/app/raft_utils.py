import os
import sys
import datetime

sys.path.append("RAFT/core")

from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

from raft import RAFT
from utils import flow_viz


class RAFTArgs:
    def __init__(self):
        self.model = "models/raft-sintel.pth"
        self.iters = 12
        self.small = True
        self.mixed_precision = True

        self.frame_1 = None
        self.frame_2 = None


def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def vizualize_flow(flow):
    print("Progress: entering vizualize_flow")
    # permute the channels and change device is necessary
    flow = flow[0].detach().permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flow_img = flow_viz.flow_to_image(flow)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR)

    return flow_img


def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(raft_args):
    # Hack
    parser = ArgumentParser()
    parser.add_argument("--model", default="models/raft-sintel.pth")
    parser.add_argument("--dataset")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--alternate_corr", action="store_true")
    args = parser.parse_args(["--model=models/raft-sintel.pth"])

    # get the RAFT model
    model = RAFT(args)

    # load pretrained weights
    pretrained_weights = torch.load(raft_args.model, map_location=torch.device("cpu"))

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    # load frames
    frame_1 = cv2.imread(raft_args.frame_1)
    frame_2 = cv2.imread(raft_args.frame_2)

    if frame_1 is None:
        print(f"Error reading image from {raft_args.frame_1}")
        return False

    if frame_2 is None:
        print(f"Error reading image from {raft_args.frame_2}")
        return False

    # resize for raft
    frame_1 = cv2.resize(frame_1, (640, 360))
    frame_2 = cv2.resize(frame_2, (640, 360))
    print("Progress: cv2.resize done")

    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)
    frame_2 = frame_preprocess(frame_2, device)
    print("Progress: frame_preprocess done")

    # perform inference
    flow_low, flow_up = model(frame_1, frame_2, iters=raft_args.iters, test_mode=True)
    print("Progress: model processing done")

    flow_img = vizualize_flow(flow_up)
    print("Progress: vizualize_flow done")

    return flow_img


if __name__ == "__main__":
    args = RAFTArgs()
    args.frame_1 = "input_frame_1.png"
    args.frame_2 = "input_frame_2.png"
    flow_img = inference(args)

    # save the image
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, flow_img)
