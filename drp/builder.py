from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairwise_euclidean_distances(features: np.ndarray) -> np.ndarray:
    """Compute all pairwise Euclidean distances without an extra dependency."""
    squared_norms = np.sum(np.square(features), axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * features @ features.T
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return np.sqrt(squared_distances, out=squared_distances)


def extract_dynamic_subgraphs(
    weighted_adjacency: np.ndarray,
    num_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select each center's strongest dynamically related neighbors."""
    num_nodes = weighted_adjacency.shape[0]
    if num_nodes <= num_neighbors:
        raise ValueError(
            f"At least {num_neighbors + 1} nodes are required, got {num_nodes}."
        )
    subgraph_size = num_neighbors + 1
    subgraphs = np.empty(
        (num_nodes, subgraph_size, subgraph_size), dtype=np.float32
    )
    node_indices = np.empty((num_nodes, subgraph_size), dtype=np.int32)

    selection_scores = weighted_adjacency.copy()
    np.fill_diagonal(selection_scores, -np.inf)
    # Stable sorting resolves equal weights by the original node index.
    neighbors = np.argsort(-selection_scores, axis=1, kind="stable")[
        :, :num_neighbors
    ]
    node_indices[:, 0] = np.arange(num_nodes)
    node_indices[:, 1:] = neighbors
    subgraphs[:] = weighted_adjacency[
        node_indices[:, :, None], node_indices[:, None, :]
    ]

    return subgraphs, node_indices


class DRPBuilder:

    def __init__(
        self,
        source_data_path: str | Path,
        *,
        alpha: float = 0.5,
        num_neighbors: int = 8,
        epsilon: float = 1e-6,
        progress_every: int = 100,
    ) -> None:
        self.source_data_path = Path(source_data_path)
        self.alpha = float(alpha)
        self.num_neighbors = int(num_neighbors)
        self.epsilon = float(epsilon)
        self.progress_every = int(progress_every)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if self.num_neighbors < 1:
            raise ValueError("num_neighbors must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.progress_every < 0:
            raise ValueError("progress_every cannot be negative.")

        self.traffic = np.load(self.source_data_path / "data.npy", mmap_mode="r")
        self.distances = np.loadtxt(self.source_data_path / "node_dist.txt")

        self._validate_inputs()
        self.num_timestamps, self.num_nodes = self.traffic.shape[:2]

        self.distance_sigma = float(np.std(self.distances))
        distance_denominator = max(self.distance_sigma, self.epsilon) ** 2
        self.distance_similarity = np.exp(
            -np.square(self.distances, dtype=np.float64) / distance_denominator
        )
        np.fill_diagonal(self.distance_similarity, 0.0)

    def _validate_inputs(self) -> None:
        if self.traffic.ndim < 3:
            raise ValueError("data.npy must have shape (time, node, ...).")

        num_nodes = self.traffic.shape[1]
        expected_distance_shape = (num_nodes, num_nodes)
        if self.distances.shape != expected_distance_shape:
            raise ValueError(
                "node_dist.txt must have shape "
                f"{expected_distance_shape}, got {self.distances.shape}."
            )
        if not np.all(np.isfinite(self.distances)):
            raise ValueError("node_dist.txt contains non-finite values.")
        if num_nodes <= self.num_neighbors:
            raise ValueError(
                f"At least {self.num_neighbors + 1} nodes are required, got {num_nodes}."
            )

    def _weighted_adjacency(self, timestamp: int) -> np.ndarray:
        node_features = np.asarray(self.traffic[timestamp], dtype=np.float64).reshape(
            self.num_nodes, -1
        )
        flow_distances = pairwise_euclidean_distances(node_features)
        flow_sigma = float(np.std(flow_distances))
        flow_similarity = np.exp(
            -np.square(flow_distances) / max(flow_sigma, self.epsilon) ** 2
        )

        weighted_adjacency = (
            self.alpha * self.distance_similarity
            + (1.0 - self.alpha) * flow_similarity
        )
        np.fill_diagonal(weighted_adjacency, 0.0)
        return weighted_adjacency

    def build(self, output_path: str | Path | None = None) -> tuple[Path, Path]:
        output_directory = Path(output_path) if output_path else self.source_data_path
        output_directory.mkdir(parents=True, exist_ok=True)

        drp_path = output_directory / "drp_adjacency.npy"
        node_path = output_directory / "drp_node_indices.npy"
        temporary_drp_path = output_directory / "drp_adjacency.tmp.npy"
        temporary_node_path = output_directory / "drp_node_indices.tmp.npy"
        subgraph_size = self.num_neighbors + 1

        drp = np.lib.format.open_memmap(
            temporary_drp_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                self.num_timestamps,
                self.num_nodes,
                subgraph_size,
                subgraph_size,
            ),
        )
        nodes = np.lib.format.open_memmap(
            temporary_node_path,
            mode="w+",
            dtype=np.int32,
            shape=(self.num_timestamps, self.num_nodes, subgraph_size),
        )

        for timestamp in range(self.num_timestamps):
            weighted_adjacency = self._weighted_adjacency(timestamp)
            drp[timestamp], nodes[timestamp] = extract_dynamic_subgraphs(
                weighted_adjacency,
                self.num_neighbors,
            )
            if self.progress_every and (
                (timestamp + 1) % self.progress_every == 0
                or timestamp + 1 == self.num_timestamps
            ):
                print(
                    f"DRP: {timestamp + 1}/{self.num_timestamps} timestamps",
                    flush=True,
                )

        drp.flush()
        nodes.flush()
        del drp, nodes
        temporary_drp_path.replace(drp_path)
        temporary_node_path.replace(node_path)

        metadata = {
            "method": "W(t) = alpha * D + (1 - alpha) * F(t)",
            "alpha": self.alpha,
            "num_neighbors": self.num_neighbors,
            "num_nodes": self.num_nodes,
            "num_timestamps": self.num_timestamps,
            "distance_sigma": self.distance_sigma,
            "self_loops": False,
            "neighbor_scope": "all non-center nodes",
            "causal_use": "Use DRP[t-1] when predicting traffic at t",
            "source_data_path": str(Path("data") / self.source_data_path.name),
            "source_sha256": {
                "data.npy": file_sha256(self.source_data_path / "data.npy"),
                "node_dist.txt": file_sha256(
                    self.source_data_path / "node_dist.txt"
                ),
            },
        }
        with (output_directory / "drp_config.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        return drp_path, node_path
