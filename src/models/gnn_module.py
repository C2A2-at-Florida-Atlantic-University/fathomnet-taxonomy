#!/usr/bin/env python3
"""
Graph Neural Network module for multi-specimen relationship modeling.

When multiple marine specimens appear in the same image, their identities
are correlated — a clear specimen can guide classification of a blurry one
nearby. This module implements a lightweight 2-layer GCN that operates on
per-specimen features grouped by image.

No torch_geometric dependency: graphs are tiny (max ~23 nodes per image),
so we implement GCN convolution directly with dense adjacency matrices.

Architecture:
    Input:  (B, D) specimen features + (B,) image_ids
    Group:  specimens by image_id → per-image subgraphs
    GCN:    2-layer GCN with gated residual per subgraph
    Output: (B, D) updated features in original order

The gated residual starts near-identity (gate bias=-2.0 → sigmoid≈0.12),
so the GNN has minimal effect early in training and gradually learns
to incorporate neighbor information.
"""

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional layer (Kipf & Welling 2017).

    H' = A_hat @ H @ W + b

    where A_hat = D^{-1/2} (A + I) D^{-1/2} is the symmetrically
    normalized adjacency with self-loops.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, D_in)
            adj_hat: Normalized adjacency (N, N) — precomputed A_hat

        Returns:
            Updated features (N, D_out)
        """
        # H @ W: linear transform
        support = x @ self.weight  # (N, D_out)
        # A_hat @ (H @ W): neighborhood aggregation
        out = adj_hat @ support  # (N, D_out)
        return out + self.bias


class SpecimenGNN(nn.Module):
    """
    2-layer GCN with gated residual for specimen relationship modeling.

    Layer 1: h1 = LayerNorm(GELU(GCNLayer(x, adj))) + x
    Layer 2: h2 = LayerNorm(GELU(GCNLayer(h1, adj))) + h1
    Gate:    g  = sigmoid(Linear([x; h2]))
    Output:  y  = g * h2 + (1-g) * x

    The gate is initialized with bias=-2.0 so sigmoid(-2)≈0.12,
    making the output start close to the original features.
    Single-node graphs (N=1) short-circuit to identity.
    """

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.gcn1 = GCNLayer(feature_dim, feature_dim)
        self.gcn2 = GCNLayer(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.act = nn.GELU()

        # Gated residual: gate = sigmoid(Linear([original; gnn_output]))
        self.gate = nn.Linear(feature_dim * 2, feature_dim)
        # Initialize gate bias to -2.0 so output starts near-identity
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, D)
            adj_hat: Normalized adjacency (N, N)

        Returns:
            Updated features (N, D)
        """
        n = x.size(0)
        if n <= 1:
            # Single node — no neighbors to aggregate, return as-is
            return x

        # Layer 1 with residual
        h1 = self.norm1(self.act(self.gcn1(x, adj_hat))) + x
        # Layer 2 with residual
        h2 = self.norm2(self.act(self.gcn2(h1, adj_hat))) + h1

        # Gated combination: learn how much GNN output to trust
        gate_input = torch.cat([x, h2], dim=1)  # (N, 2D)
        g = torch.sigmoid(self.gate(gate_input))  # (N, D)
        return g * h2 + (1 - g) * x


def build_fully_connected_adj_hat(n: int, device: torch.device) -> torch.Tensor:
    """
    Build symmetrically normalized adjacency for a fully-connected graph.

    For a fully-connected graph of N nodes with self-loops:
    A + I = ones(N, N), so D = N*I, and
    A_hat = D^{-1/2}(A+I)D^{-1/2} = (1/N) * ones(N, N)

    Args:
        n: Number of nodes
        device: Torch device

    Returns:
        A_hat: (N, N) normalized adjacency
    """
    return torch.full((n, n), 1.0 / n, device=device)


def apply_gnn_to_batch(
    gnn: SpecimenGNN,
    features: torch.Tensor,
    image_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Apply GNN to batch of specimens, grouped by image.

    Groups specimens by image_id, applies GNN independently to each
    image's subgraph (fully connected), then reassembles results in
    original batch order.

    Args:
        gnn: SpecimenGNN module
        features: (B, D) specimen features
        image_ids: (B,) integer image IDs for grouping

    Returns:
        Updated features (B, D) in same order as input
    """
    device = features.device
    input_dtype = features.dtype
    B, D = features.shape
    output = torch.empty_like(features)

    # Group indices by image_id
    unique_ids = image_ids.unique()

    for img_id in unique_ids:
        mask = image_ids == img_id
        indices = mask.nonzero(as_tuple=True)[0]
        n = indices.size(0)

        # Extract subgraph features
        sub_features = features[indices]  # (n, D)

        if n == 1:
            # Single specimen — GNN returns identity
            output[indices] = sub_features
        else:
            # Build fully-connected normalized adjacency
            adj_hat = build_fully_connected_adj_hat(n, device)
            # Apply GNN (may upcast to float32 under AMP)
            updated = gnn(sub_features, adj_hat)  # (n, D)
            output[indices] = updated.to(input_dtype)

    return output
