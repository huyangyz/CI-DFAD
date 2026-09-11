"""CI-DFAD model defined in the same order as the paper framework."""

import torch
import torch.nn as nn

from bayesian import BayesLinear
from layers import LongTermNonlinearEncoder, TwoLayerNonlinearEncoder


ARCHITECTURE_VERSION = 3


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, activation="sigmoid"):
        super().__init__()
        self.projection = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh() if activation == "tanh" else nn.Sigmoid()

    def forward(self, features, adjacency):
        return self.activation(self.projection(torch.bmm(adjacency, features)))


class GCGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        gate_input_size = input_size + hidden_size
        self.reset_gate = GraphConvolution(gate_input_size, hidden_size)
        self.update_gate = GraphConvolution(gate_input_size, hidden_size)
        self.candidate = GraphConvolution(
            gate_input_size, hidden_size, activation="tanh"
        )

    def forward(self, features, hidden_state, adjacency):
        combined = torch.cat([features, hidden_state], dim=2)
        reset = self.reset_gate(combined, adjacency)
        update = self.update_gate(combined, adjacency)
        candidate_input = torch.cat([features, reset * hidden_state], dim=2)
        candidate = self.candidate(candidate_input, adjacency)
        return update * hidden_state + (1.0 - update) * candidate


class GCGRUEncoder(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.num_layers = options["num_layer"]
        self.num_nodes = options["num_adj"]
        self.hidden_size = options["hidden_dim"]
        self.layers = nn.ModuleList(
            GCGRUCell(
                options["num_feature"] if layer == 0 else self.hidden_size,
                self.hidden_size,
            )
            for layer in range(self.num_layers)
        )

    def forward(self, sequence, adjacency):
        hidden_state = None
        output = None
        for time_step in range(sequence.shape[1]):
            output, hidden_state = self.encode_step(
                sequence[:, time_step], adjacency, hidden_state
            )
        return output, hidden_state

    def encode_step(self, features, adjacency, hidden_state=None):
        if hidden_state is None:
            hidden_state = features.new_zeros(
                self.num_layers,
                features.shape[0],
                self.num_nodes,
                self.hidden_size,
            )
        layer_states = []
        output = features
        for layer_index, layer in enumerate(self.layers):
            output = layer(output, hidden_state[layer_index], adjacency)
            layer_states.append(output)
        return output, torch.stack(layer_states)


class CosineSpatialAttention(nn.Module):

    def __init__(self, num_nodes):
        super().__init__()
        self.learnable_weights = nn.Parameter(
            torch.randn(1, 1, num_nodes, num_nodes)
        )

    def forward(self, sequence):
        norms = torch.norm(sequence, p=2, dim=-1, keepdim=True)
        dot_products = torch.matmul(sequence, sequence.transpose(-1, -2))
        norm_products = torch.matmul(norms, norms.transpose(-1, -2))
        cosine_similarity = dot_products / (norm_products + 1e-7)
        attention = torch.softmax(
            self.learnable_weights * cosine_similarity, dim=-1
        )
        return torch.matmul(attention, sequence)


class CoGRU(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.cosine_attention = CosineSpatialAttention(options["num_adj"])
        self.recurrent_encoder = GCGRUEncoder(options)

    def forward(self, sequence, adjacency):
        attended_sequence = self.cosine_attention(sequence)
        return self.recurrent_encoder(attended_sequence, adjacency)


class WeightedGateFusion(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.gate_projection = nn.Linear(feature_size * 2, feature_size)

    def forward(self, lower_scale, higher_scale):
        gate = torch.sigmoid(
            self.gate_projection(torch.cat([lower_scale, higher_scale], dim=-1))
        )
        return gate * lower_scale + (1.0 - gate) * higher_scale


class CrossScaleFeatureInteraction(nn.Module):

    def __init__(self, feature_size):
        super().__init__()
        self.hour_day_fusion = WeightedGateFusion(feature_size)
        self.day_week_fusion = WeightedGateFusion(feature_size)

    def forward(self, hourly_features, daily_features, weekly_features):
        hour_day_interaction = self.hour_day_fusion(hourly_features, daily_features)
        day_week_interaction = self.day_week_fusion(daily_features, weekly_features)
        daily_enhanced = hour_day_interaction + daily_features
        weekly_enhanced = day_week_interaction + weekly_features
        return daily_enhanced, weekly_enhanced


class UncertaintyAwareModeling(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.bayesian_projection = BayesLinear(
            prior_mu=0,
            prior_sigma=0.1,
            in_features=input_size,
            out_features=output_size,
        )
        self.activation = nn.Tanh()

    def forward(self, features, adjacency):
        aggregated = torch.bmm(adjacency, features)
        return self.activation(self.bayesian_projection(aggregated))


class Generator(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        hidden_size = options["hidden_dim"]
        long_term_hidden_sizes = [hidden_size, hidden_size]

        self.cogru = CoGRU(options)
        self.daily_encoder = LongTermNonlinearEncoder(
            input_dim=options["num_feature"],
            hidden_layer_sizes=long_term_hidden_sizes,
            output_dim=hidden_size,
        )
        self.weekly_encoder = LongTermNonlinearEncoder(
            input_dim=options["num_feature"],
            hidden_layer_sizes=long_term_hidden_sizes,
            output_dim=hidden_size,
        )
        self.external_encoder = TwoLayerNonlinearEncoder(
            [options["time_feature"], hidden_size, hidden_size]
        )
        self.cfi_re = CrossScaleFeatureInteraction(hidden_size)
        self.uam = UncertaintyAwareModeling(hidden_size * 4, options["num_feature"])

    def forward(self, hourly_data, weekly_data, daily_data, adjacency, time_features):
        batch_size = hourly_data.shape[0]
        hourly_features, _ = self.cogru(hourly_data, adjacency)
        daily_features = self.daily_encoder(daily_data).view(batch_size, 1, -1)
        weekly_features = self.weekly_encoder(weekly_data).view(batch_size, 1, -1)
        external_features = self.external_encoder(time_features).view(
            batch_size, 1, -1
        )

        daily_features = daily_features.repeat(1, self.options["num_adj"], 1)
        weekly_features = weekly_features.repeat(1, self.options["num_adj"], 1)
        external_features = external_features.repeat(1, self.options["num_adj"], 1)
        daily_enhanced, weekly_enhanced = self.cfi_re(
            hourly_features, daily_features, weekly_features
        )
        combined_features = torch.cat(
            [hourly_features, daily_enhanced, weekly_enhanced, external_features],
            dim=2,
        )
        return self.uam(combined_features, adjacency)


class Discriminator(nn.Module):
    def __init__(self, options):
        super().__init__()
        hidden_size = options["hidden_dim"]
        self.spatial_encoder = GraphConvolution(options["num_feature"], hidden_size)
        self.temporal_encoder = GCGRUEncoder(options)
        self.temporal_projection = nn.Sequential(
            nn.Linear(hidden_size * options["num_adj"], hidden_size),
            nn.ReLU(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, sequence, adjacency):
        temporal_features, _ = self.temporal_encoder(sequence[:, :-1], adjacency)
        temporal_features = self.temporal_projection(
            temporal_features.reshape(sequence.shape[0], -1)
        )
        spatial_features = self.spatial_encoder(sequence[:, -1], adjacency)
        spatial_features = torch.max(spatial_features, dim=1).values
        return self.output_projection(
            torch.cat([spatial_features, temporal_features], dim=1)
        )
