"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import rich
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import seaborn as sns
import sklearn.metrics

import models as models
import losses
import utils
import dataloaders
import evaluators
from utils import sliding_windows
from dataloaders import MOOD2dDataSet, BRAIN_DEFAULT_TRANSFORM, ABDOM_DEFAULT_TRANSFORM
from torch.utils.data import DataLoader
LOGGER = logging.getLogger()

   
def write_results(filepaths, list_, save_dir):    
    for f, l in zip(filepaths, list_):
        base = f[0].split("/")[-1]

        savepath = save_dir + base + ".txt"
        with open(savepath, "w") as file:
            # Loop through the list and write each element to the file
            file.write(str(l))
            
def compute_results(filenames, score, save_dir, scan):   
    test_probs = np.max(np.max(np.array(score), axis=1), axis=1)

    # Rescale
    min_value, max_value = test_probs.min(), test_probs.max()
    test_probs = (test_probs - min_value) / (max_value - min_value)

    y_score = test_probs.tolist()
    write_results(filenames, y_score, save_dir)
                      

def get_elbos(dataloader, model, save_dir, scan):
    with torch.no_grad():
        fn_names = []
        elbos = []

        n = 0

        # load batch of slices (1 ct) - brain 256, abdom 512
        for b, data in enumerate(dataloader):
            x_all, fn = data
            x_all = torch.squeeze(x_all)
            if scan == "brain":
                x_all = sliding_windows(x_all, 32, 8)
            elif scan == "abdom":
                x_all = sliding_windows(x_all, 32, 32)

            n += 1

            fn_names.append(fn)
            window_elbos = []

            for window in range(x_all.shape[1]):
                x = x_all[:,window,:,:].to(device)

                sample_elbos = []

                # Regular ELBO
                for i in range(args.iw_samples_elbo):
                    likelihood_data, stage_datas = model(x, decode_from_p=False, use_mode=False)
                    kl_divergences = [
                        stage_data.loss.kl_elementwise
                        for stage_data in stage_datas
                        if stage_data.loss.kl_elementwise is not None
                    ]

                    loss, elbo, likelihood, kl_divergences = criterion(
                        likelihood_data.likelihood,
                        kl_divergences,
                        samples=1,
                        free_nats=0,
                        beta=1,
                        sample_reduction=None,
                        batch_reduction=None,
                    )
                    sample_elbos.append(elbo.detach())

                # Compile ELBO samples into Tensor
                sample_elbos = torch.stack(sample_elbos, axis=0) # slice, 1

                # Apply logsumexp to ELBO tensor
                sample_elbo = utils.log_sum_exp(sample_elbos, axis=0)

                # Compile likelihood samples into tensor
                window_elbos.append(sample_elbo) # Append - collect across sliding window

            window_elbos = torch.stack(window_elbos, axis=1) # Compile values from sliding window into tensor (256, 16)
            elbos.append(window_elbos.tolist()) # Collect results for each CT

        compute_results(fn_names, elbos, save_dir, scan)


if __name__ == "__main__":

    # models/MOOD128x128Dequantized-2023-08-27-19-51-06.682507
    # BrainMOOD128x128Dequantized-2023-08-28-20-17-37.529063/
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/worskspace/models/BrainMOOD128x128Dequantized-2023-08-28-20-17-37.529063/", help="model")
    parser.add_argument("--scan", type=str, default="brain", help="Which scan - brain or abdom")
    parser.add_argument("--iw_samples_elbo", type=int, default=3, help="importances samples for regular ELBO")
    parser.add_argument("--iw_samples_Lk", type=int, default=1, help="importances samples for L>k bound")
    parser.add_argument("--n_eval_examples", type=int, default=float("inf"), help="cap on the number of examples to use")
    parser.add_argument("--n_latents_skip", type=int, default=0, help="the value of k in the paper")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="device to evaluate on")
    parser.add_argument("--save_dir", type=str, default="", help="directory to store scores in")
    parser.add_argument("--source_dir", type=str, default="", help="directory from which to load scores")

    args = parser.parse_args()
    rich.print(vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define checkpoints and load model
    checkpoint = models.Checkpoint(path=args.model_dir)
    checkpoint.load()

    model = checkpoint.model
    model.eval()
    model.to(device)

    criterion = losses.ELBO()

    if args.scan == "brain":
        TRANSFORM = BRAIN_DEFAULT_TRANSFORM
    elif args.scan == "abdom":
        TRANSFORM = ABDOM_DEFAULT_TRANSFORM

    dataset = MOOD2dDataSet(
        root=args.source_dir,
        transform=TRANSFORM,
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    get_elbos(dataloader, model, save_dir=args.save_dir, scan=args.scan)