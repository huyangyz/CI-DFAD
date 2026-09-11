from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, options):
        self.options = options
        source_path = Path(options["source_data_path"])
        processed_path = Path(options["processed_data_path"])

        self.traffic = torch.from_numpy(np.load(source_path / "data.npy")).float()
        self.time_features = torch.from_numpy(
            np.loadtxt(source_path / "time_features.txt")
        ).float()
        self.drp_adjacency = torch.from_numpy(
            np.load(processed_path / "drp_adjacency.npy", mmap_mode="c")
        ).float()
        self.drp_node_indices = torch.from_numpy(
            np.load(processed_path / "drp_node_indices.npy", mmap_mode="c")
        )
        self.total_time = min(
            self.traffic.shape[0],
            self.time_features.shape[0],
            self.drp_adjacency.shape[0],
            self.drp_node_indices.shape[0],
        )
        self.recent_steps = options["recent_time"] * options["timestamp"]
        self.trend_steps = options["trend_time"] * options["timestamp"]
        self.daily_steps = options["day_time"] * options["timestamp"]
        self.start_time = (
            self.trend_steps if options["is_train"] else options["train_time"]
        )
        end_time = options["train_time"] if options["is_train"] else self.total_time
        if not self.start_time < end_time <= self.total_time:
            raise ValueError(
                f"Invalid time split [{self.start_time}, {end_time}) for "
                f"{self.total_time} aligned time steps."
            )

        self.time_count = end_time - self.start_time
        self.input_size = self.traffic.shape[2] * self.traffic.shape[3]
        self.subgraph_size = self.drp_node_indices.shape[2]
        self.node_count = self.traffic.shape[1]
        self._normalize_traffic()
        self.length = self.node_count * self.time_count

    def __getitem__(self, index):
        target_time = index // self.node_count + self.start_time
        center_node = index % self.node_count

        # Causal DRP: the target X_t must not be used to construct its own graph.
        graph_time = target_time - 1
        node_indices = self.drp_node_indices[graph_time, center_node].long()
        hourly_data = self.traffic[
            target_time - self.recent_steps : target_time, node_indices
        ].reshape(self.recent_steps, self.subgraph_size, self.input_size)
        target = self.traffic[target_time, node_indices].reshape(
            self.subgraph_size, self.input_size
        )
        weekly_data = self.traffic[
            target_time - self.trend_steps : target_time, center_node
        ].reshape(self.trend_steps, self.input_size)
        daily_data = self.traffic[
            target_time - self.daily_steps : target_time, center_node
        ].reshape(self.daily_steps, self.input_size)
        adjacency = self.normalize_adjacency(
            self.drp_adjacency[graph_time, center_node]
        )
        return (
            (hourly_data, weekly_data, daily_data, self.time_features[target_time]),
            adjacency,
            target,
            target_time - self.start_time,
            center_node,
        )

    @staticmethod
    def normalize_adjacency(adjacency):
        adjacency = adjacency + torch.eye(
            adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
        )
        inverse_sqrt_degree = (adjacency.sum(dim=1) + 1e-5).pow(-0.5)
        return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]

    def _normalize_traffic(self):
        training_data = self.traffic[: self.options["train_time"]]
        for source in range(self.traffic.shape[-1]):
            maximum = training_data[..., source].max()
            minimum = training_data[..., source].min()
            if maximum == minimum:
                raise ValueError(f"Traffic source {source} is constant in training")
            self.traffic[..., source] = (
                2 * (self.traffic[..., source] - minimum) / (maximum - minimum) - 1
            )

    def __len__(self):
        return self.length
