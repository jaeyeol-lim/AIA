"""AIA model adapted from categorical OGB features to DrugOOD dense features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, global_add_pool


class MaskedGINConv(MessagePassing):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        edge_embedding = self.edge_encoder(edge_attr.float())
        messages = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_embedding,
            edge_weight=edge_weight,
        )
        return self.mlp((1 + self.eps) * x + messages)

    def message(
        self,
        x_j: Tensor,
        edge_attr: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        message = x_j + edge_attr
        if edge_weight is not None:
            message = message * edge_weight
        return F.relu(message)


class DrugOODGINEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        edge_dim: int,
        dropout: float,
        node_dim: int | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim) if node_dim is not None else None
        self.convs = nn.ModuleList(
            [MaskedGINConv(hidden_dim, edge_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.residual = residual

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        node_weight: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if self.node_encoder is not None:
            x = self.node_encoder(x.float())
        if node_weight is not None:
            x = x * node_weight
        for index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            previous = x
            x = norm(conv(x, edge_index, edge_attr, edge_weight))
            if index + 1 < len(self.convs):
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + previous
        return x


def scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, values.shape[-1]))
    return output.index_add(0, index, values)


class GraphMasker(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = DrugOODGINEncoder(
            num_layers=num_layers,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.node_attention = nn.Linear(hidden_dim, 1)
        self.edge_attention = nn.Linear(2 * hidden_dim, 1)

    def forward(self, batch) -> dict[str, Tensor]:
        node_repr = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        row, col = batch.edge_index
        node_key = torch.sigmoid(self.node_attention(node_repr))
        edge_key = torch.sigmoid(
            self.edge_attention(torch.cat((node_repr[row], node_repr[col]), dim=-1))
        )
        num_graphs = int(batch.num_graphs)
        node_key_num, node_env_num, node_nonzero = self._mask_stats(
            node_key, batch.batch, num_graphs
        )
        edge_batch = batch.batch[row]
        edge_key_num, edge_env_num, edge_nonzero = self._mask_stats(
            edge_key, edge_batch, num_graphs
        )
        return {
            "node_key": node_key,
            "edge_key": edge_key,
            "node_key_num": node_key_num,
            "node_env_num": node_env_num,
            "edge_key_num": edge_key_num,
            "edge_env_num": edge_env_num,
            "node_nonzero": node_nonzero,
            "edge_nonzero": edge_nonzero,
        }

    @staticmethod
    def _mask_stats(mask: Tensor, graph_index: Tensor, num_graphs: int):
        key_num = scatter_sum(mask, graph_index, num_graphs) + 1e-8
        env_num = scatter_sum(1 - mask, graph_index, num_graphs) + 1e-8
        nonzero = scatter_sum((mask > 0).float(), graph_index, num_graphs)
        total = scatter_sum(torch.ones_like(mask), graph_index, num_graphs)
        return key_num, env_num, nonzero / (total + 1e-8)


class AIADrugOOD(nn.Module):
    """Original AIA causal/attacker structure with a 2+2 layer DrugOOD GIN."""

    def __init__(
        self,
        node_dim: int = 39,
        edge_dim: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        stable_feature_ratio: float = 0.5,
        adv_node_ratio: float = 1.0,
        adv_edge_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.stable_feature_ratio = stable_feature_ratio
        self.adv_node_ratio = adv_node_ratio
        self.adv_edge_ratio = adv_edge_ratio
        self.front = DrugOODGINEncoder(
            num_layers=2,
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.back = DrugOODGINEncoder(
            num_layers=2,
            node_dim=None,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.causaler = GraphMasker(node_dim, edge_dim, hidden_dim, 2, dropout)
        self.attacker = GraphMasker(node_dim, edge_dim, hidden_dim, 2, dropout)
        self.predictor = nn.Linear(hidden_dim, 2)
        self.distance = nn.MSELoss()

    def encode(self, batch, node_weight=None, edge_weight=None) -> Tensor:
        x = self.front(batch.x, batch.edge_index, batch.edge_attr)
        return self.encode_back(batch, x, node_weight, edge_weight)

    def encode_back(self, batch, x, node_weight=None, edge_weight=None) -> Tensor:
        x = self.back(
            x,
            batch.edge_index,
            batch.edge_attr,
            node_weight=node_weight,
            edge_weight=edge_weight,
        )
        return global_add_pool(x, batch.batch)

    def forward_erm(self, batch) -> Tensor:
        return self.predictor(self.encode(batch))

    def forward_causal(self, batch) -> Tensor:
        causal = self.causaler(batch)
        return self.predictor(
            self.encode(batch, causal["node_key"], causal["edge_key"])
        )

    def forward_advcausal(self, batch) -> dict[str, Tensor]:
        causal = self.causaler(batch)
        adversarial = self.attacker(batch)
        front = self.front(batch.x, batch.edge_index, batch.edge_attr)
        node_combined = (1 - causal["node_key"]) * adversarial["node_key"] + causal["node_key"]
        edge_combined = (1 - causal["edge_key"]) * adversarial["edge_key"] + causal["edge_key"]
        return {
            "pred_causal": self.predictor(
                self.encode_back(batch, front, causal["node_key"], causal["edge_key"])
            ),
            "pred_combined": self.predictor(
                self.encode_back(batch, front, node_combined, edge_combined)
            ),
            "causal_regularizer": self._regularizer(
                causal, self.stable_feature_ratio, self.stable_feature_ratio
            ),
        }

    def forward_attack(self, batch) -> dict[str, Tensor]:
        adversarial = self.attacker(batch)
        front = self.front(batch.x, batch.edge_index, batch.edge_attr)
        original_repr = self.encode_back(batch, front)
        adversarial_repr = self.encode_back(
            batch, front, adversarial["node_key"], adversarial["edge_key"]
        )
        return {
            "pred_adversarial": self.predictor(adversarial_repr),
            "distance": self.distance(adversarial_repr, original_repr),
            "adversarial_regularizer": self._regularizer(
                adversarial, self.adv_node_ratio, self.adv_edge_ratio
            ),
        }

    @staticmethod
    def _ratio_regularizer(key: Tensor, env: Tensor, ratio: float, nonzero: Tensor) -> Tensor:
        target = torch.full_like(key, ratio)
        # Keep the official AIA regularizer form for reproducibility.
        return (key / (key + env) - target).abs().mean() + (nonzero - target).mean()

    def _regularizer(self, masks: dict[str, Tensor], node_ratio: float, edge_ratio: float) -> Tensor:
        node = self._ratio_regularizer(
            masks["node_key_num"], masks["node_env_num"], node_ratio, masks["node_nonzero"]
        )
        edge = self._ratio_regularizer(
            masks["edge_key_num"], masks["edge_env_num"], edge_ratio, masks["edge_nonzero"]
        )
        return node + edge


def initialize_aia(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
