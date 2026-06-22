import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.use_cosine_similarity = use_cosine_similarity

    def forward(self, zis, zjs):
        # Check input shapes
        assert zis.shape[0] == zjs.shape[0] == self.batch_size, \
            f"Batch size mismatch: expected {self.batch_size}, got {zis.shape[0]}"

        # Merge representations
        representations = torch.cat([zis, zjs], dim=0)  # (2N, C)
        N = self.batch_size

        # Normalize representations if using cosine similarity
        if self.use_cosine_similarity:
            representations = F.normalize(representations, dim=1)

        # Compute similarity matrix
        similarity_matrix = (
            F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            if self.use_cosine_similarity
            else torch.matmul(representations, representations.T)
        ) 
        # Create labels for positive pairs
        labels = torch.arange(N, device=self.device).repeat(2)  # [0,1,...,N-1,0,1,...,N-1]

        # Create positive pair indices (zis[i] pairs with zjs[i] and vice versa)
        pos_indices = torch.arange(2 * N, device=self.device)
        pos_pairs = torch.stack([
            pos_indices,  # zis[0], zjs[0], ..., zis[N-1], zjs[N-1]
            torch.cat([torch.arange(N, 2 * N, device=self.device), torch.arange(N, device=self.device)])  # zjs[0], zis[0], ..., zjs[N-1], zis[N-1]
        ], dim=1)  # Shape: (2N, 2)

        # Extract positive similarities
        pos_sim = similarity_matrix[pos_pairs[:, 0], pos_pairs[:, 1]].view(2 * N, 1)

        # Use similarity matrix as logits, with positive pairs on the diagonal
        loss = self.criterion(similarity_matrix, labels)

        return loss