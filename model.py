import numpy as np
import torch
import torch.nn as nn
from collections import deque

# NTU-25: 6=left_wrist, 10=right_wrist
_HAND_JOINT_INDICES = (6, 10)


class Graph:
    def __init__(self, num_nodes, strategy="spatial", interaction_mode="full"):
        """
        interaction_mode:
          - "none":       two independent skeletons only
          - "full":       every P1 joint <-> every P2 joint (dense; 625 cross edges)
          - "hand_cross": each wrist linked to all joints on the other person (both
                          directions) — better for combat sports, cheaper adjacency
        """
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.interaction_mode = interaction_mode

        self.nodes_per_person = 25  # NTU-25
        self.center = 0             # Base of spine

        self.edges = self.get_edges()
        self.A = self.get_adjacency_matrix()

    def get_edges(self):
        """NTU-25 physical edges + optional cross-person edges."""
        p1_edges = [
            (0, 1), (1, 20), (20, 2), (2, 3),         # Spine + head
            (20, 4), (4, 5), (5, 6), (6, 7),           # Left arm
            (7, 21), (7, 22),                          # Left hand tip + thumb
            (20, 8), (8, 9), (9, 10), (10, 11),        # Right arm
            (11, 23), (11, 24),                        # Right hand tip + thumb
            (0, 12), (12, 13), (13, 14), (14, 15),     # Left leg
            (0, 16), (16, 17), (17, 18), (18, 19),     # Right leg
        ]

        if self.num_nodes == self.nodes_per_person:
            return p1_edges

        p2_edges = [(u + self.nodes_per_person, v + self.nodes_per_person)
                    for u, v in p1_edges]
        all_edges = p1_edges + p2_edges

        p2_off = self.nodes_per_person
        if self.interaction_mode == "full":
            for i in range(self.nodes_per_person):
                for j in range(self.nodes_per_person):
                    all_edges.append((i, j + p2_off))
        elif self.interaction_mode == "hand_cross":
            # Only wrists cross the person boundary — more principled for karate
            for hi in _HAND_JOINT_INDICES:
                for j in range(self.nodes_per_person):
                    all_edges.append((hi, j + p2_off))
            for hi in _HAND_JOINT_INDICES:
                for i in range(self.nodes_per_person):
                    all_edges.append((hi + p2_off, i))

        return all_edges

    def get_adjacency_matrix(self):
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i, j in self.edges:
            adj[i, j] = 1
            adj[j, i] = 1

        dist_from_center = np.full(self.num_nodes, np.inf, dtype=np.float32)
        centers = [self.center]
        if self.num_nodes > self.nodes_per_person:
            centers.append(self.center + self.nodes_per_person)

        for c in centers:
            if c < self.num_nodes:
                dist_from_center[c] = 0
                q = deque([c])
                while q:
                    curr = q.popleft()
                    for neighbor in range(self.num_nodes):
                        if adj[curr, neighbor] == 1 and dist_from_center[neighbor] == np.inf:
                            dist_from_center[neighbor] = dist_from_center[curr] + 1
                            q.append(neighbor)

        num_partitions = 3
        A = np.zeros((num_partitions, self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    A[0, i, j] = 1
                elif adj[i, j] == 1:
                    if dist_from_center[j] < dist_from_center[i]:
                        A[1, i, j] = 1
                    else:
                        A[2, i, j] = 1

        for p in range(num_partitions):
            row_sums = A[p].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A[p] = A[p] / row_sums

        return torch.FloatTensor(A)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subsets = A.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, kernel_size=1)
        self.register_buffer('A', A)
        self.edge_importance = nn.Parameter(torch.ones(self.A.size()))

    def forward(self, x):
        N, C, T, V = x.size()
        x = self.conv(x)
        x = x.view(N, self.num_subsets, -1, T, V)
        adj = self.A * self.edge_importance
        x = torch.einsum('nkctv,kvw->nctw', x, adj)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.5):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            # Dropout AFTER the full TCN BN, not sandwiched between BNs
            nn.Dropout(dropout),
        )
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """
    9-layer ST-GCN following the channel progression of the original paper:
        64  x3  (layers 1-3)   — low-level spatial feature extraction
        128 x3  (layers 4-6)   — mid-level motion patterns; stride=2 at layer 4
        256 x3  (layers 7-9)   — high-level temporal composition; stride=2 at layer 7

    This gives two temporal downsampling steps, expanding the effective temporal
    receptive field to cover longer action sequences.

    Args
    ----
    num_classes       : number of action classes
    in_channels       : input feature channels (typically 2 for x,y or 3 for x,y,conf)
    num_nodes         : total graph nodes (50 for two-person NTU-25)
    interaction_mode  : "full" (default) | "hand_cross" | "none"
    dropout           : dropout probability applied inside each TCN block
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        num_nodes,
        interaction_mode="full",
        dropout=0.5,
    ):
        super().__init__()

        graph = Graph(num_nodes, interaction_mode=interaction_mode)
        A = graph.A  # (3, V, V)

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 64-channel stage 
        self.layer1 = STGCNBlock(in_channels, 64, A, dropout=dropout)
        self.layer2 = STGCNBlock(64,          64, A, dropout=dropout)
        self.layer3 = STGCNBlock(64,          64, A, dropout=dropout)

        # 128-channel stage  (stride=2 on layer 4) 
        self.layer4 = STGCNBlock(64,  128, A, stride=2, dropout=dropout)
        self.layer5 = STGCNBlock(128, 128, A, dropout=dropout)
        self.layer6 = STGCNBlock(128, 128, A, dropout=dropout)

        # 256-channel stage  (stride=2 on layer 7) 
        self.layer7 = STGCNBlock(128, 256, A, stride=2, dropout=dropout)
        self.layer8 = STGCNBlock(256, 256, A, dropout=dropout)
        self.layer9 = STGCNBlock(256, 256, A, dropout=dropout)

        # Output head 
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x : (N, C, T, V)
        returns logits : (N, num_classes)
        """
        N, C, T, V = x.size()

        # Input batch norm over the joint/channel dimension
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        # Global average pooling over (T, V) → (N, 256)
        x = x.mean(dim=[2, 3])

        return self.fc(x)