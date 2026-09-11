import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TrafficDataset
from model import ARCHITECTURE_VERSION, Discriminator, Generator


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("Checkpoint architecture mismatch; retrain the model.")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


class Evaluator:
    def __init__(self, options):
        self.options = options
        self.device = options["device"]
        self.dataset = TrafficDataset(options)
        self.loader = DataLoader(
            self.dataset,
            batch_size=options["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        epoch = options["epoch"]
        checkpoint_path = options["checkpoint_path"]
        self.generator = load_checkpoint(
            Generator(options),
            checkpoint_path / f"generator_epoch_{epoch:03d}.pt",
            self.device,
        ).to(self.device)
        self.discriminator = load_checkpoint(
            Discriminator(options),
            checkpoint_path / f"discriminator_epoch_{epoch:03d}.pt",
            self.device,
        ).to(self.device)

    def evaluate(self):
        self.generator.eval()
        self.discriminator.eval()
        component_scores = torch.zeros(
            (self.dataset.time_count, self.dataset.node_count, 4)
        )
        multi_view_scores = torch.zeros(
            (self.dataset.time_count, self.dataset.node_count, 3)
        )
        sample_count = self.options["mc_samples"]
        if sample_count < 2:
            raise ValueError("mc_samples must be at least 2")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()
        with torch.no_grad():
            for step, batch in enumerate(self.loader):
                (hourly, weekly, daily, external), adjacency, target, times, nodes = batch
                hourly, weekly, daily, external, adjacency, target = (
                    tensor.to(self.device)
                    for tensor in (hourly, weekly, daily, external, adjacency, target)
                )
                samples = torch.stack(
                    [
                        self.generator(hourly, weekly, daily, adjacency, external)
                        for _ in range(sample_count)
                    ]
                )
                mean_prediction = samples.mean(dim=0)
                prediction_std = samples.std(dim=0)
                reconstruction = ((mean_prediction - target) ** 2).mean(dim=(1, 2))
                uncertainty = (
                    torch.abs(target - mean_prediction) / (prediction_std + 1e-6)
                ).mean(dim=(1, 2))
                real_sequence = torch.cat([hourly, target.unsqueeze(1)], dim=1)
                generated_sequence = torch.cat(
                    [hourly, mean_prediction.unsqueeze(1)], dim=1
                )
                real_score = self.discriminator(real_sequence, adjacency).squeeze(-1)
                generated_score = self.discriminator(
                    generated_sequence, adjacency
                ).squeeze(-1)
                discriminator_difference = real_score - generated_score

                component_scores[times.long(), nodes.long()] = torch.stack(
                    [reconstruction, real_score, generated_score, uncertainty], dim=-1
                ).cpu()
                multi_view_scores[times.long(), nodes.long()] = torch.stack(
                    [reconstruction, discriminator_difference, uncertainty], dim=-1
                ).cpu()
                if step % 100 == 0:
                    logging.info("step:%d reconstruction:%.6f", step, reconstruction.mean())

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time = time.perf_counter() - start_time

        np.save(
            self.options["result_path"] / "component_scores.npy",
            component_scores.numpy(),
        )
        return inference_time
