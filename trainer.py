import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrafficDataset
from model import ARCHITECTURE_VERSION, Discriminator, Generator


class Trainer:
    def __init__(self, options):
        self.options = options
        self.device = options["device"]
        self.loader = DataLoader(
            TrafficDataset(options), batch_size=options["batch_size"], shuffle=True
        )
        self.generator = Generator(options).to(self.device)
        self.discriminator = Discriminator(options).to(self.device)
        self.reconstruction_loss = nn.MSELoss()
        self.generator_optimizer = torch.optim.RMSprop(
            self.generator.parameters(), lr=options["lr"]
        )
        self.discriminator_optimizer = torch.optim.RMSprop(
            self.discriminator.parameters(), lr=options["lr"]
        )

    def train(self):
        self.generator.train()
        self.discriminator.train()
        epoch_times = []
        for epoch in range(1, self.options["epoch"] + 1):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            for step, batch in enumerate(self.loader):
                (hourly, weekly, daily, external), adjacency, target, _, _ = batch
                hourly, weekly, daily, external, adjacency, target = (
                    tensor.to(self.device)
                    for tensor in (hourly, weekly, daily, external, adjacency, target)
                )

                real_sequence = torch.cat([hourly, target.unsqueeze(1)], dim=1)
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    generated = self.generator(
                        hourly, weekly, daily, adjacency, external
                    )
                generated_sequence = torch.cat(
                    [hourly, generated.unsqueeze(1)], dim=1
                )
                discriminator_loss = (
                    -self.discriminator(real_sequence, adjacency).mean()
                    + self.discriminator(generated_sequence, adjacency).mean()
                )
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                with torch.no_grad():
                    for parameter in self.discriminator.parameters():
                        parameter.clamp_(-0.05, 0.05)

                self.generator_optimizer.zero_grad(set_to_none=True)
                generated = self.generator(hourly, weekly, daily, adjacency, external)
                mean_squared_error = self.reconstruction_loss(generated, target)
                generated_sequence = torch.cat(
                    [hourly, generated.unsqueeze(1)], dim=1
                )
                adversarial_loss = -self.discriminator(
                    generated_sequence, adjacency
                ).mean()
                kl_per_sample = (
                    self.generator.uam.bayesian_projection.kl_loss()
                    / len(self.loader.dataset)
                )
                kan_regularization = (
                    self.generator.daily_encoder.regularization_loss()
                    + self.generator.weekly_encoder.regularization_loss()
                    + self.generator.external_encoder.regularization_loss()
                    / len(self.generator.external_encoder.layers)
                )
                reconstruction_term = self.options["lambda_G"] * mean_squared_error
                uncertainty_term = self.options["lambda_U"] * kl_per_sample
                kan_term = self.options["lambda_A"] * kan_regularization
                generator_loss = (
                    reconstruction_term
                    + adversarial_loss
                    + uncertainty_term
                    + kan_term
                )
                generator_loss.backward()
                self.generator_optimizer.step()

                if step % 100 == 0:
                    values = (
                        epoch,
                        step,
                        discriminator_loss.item(),
                        generator_loss.item(),
                        reconstruction_term.item(),
                        adversarial_loss.item(),
                        uncertainty_term.item(),
                        kan_term.item(),
                        time.perf_counter() - start_time,
                    )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            epoch_times.append(time.perf_counter() - start_time)
            self._save_checkpoint(self.generator, "generator", epoch)
            self._save_checkpoint(self.discriminator, "discriminator", epoch)
        return epoch_times

    def _save_checkpoint(self, model, model_name, epoch):
        torch.save(
            {
                "architecture_version": ARCHITECTURE_VERSION,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            },
            self.options["checkpoint_path"]
            / f"{model_name}_epoch_{epoch:03d}.pt",
        )
