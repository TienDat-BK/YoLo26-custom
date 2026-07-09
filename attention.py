import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, numHeads : int = 8, sequence_len : int = 64, emb_dim : int = 64):
        super().__init__()
        self.numHeads = numHeads
        self.seqence_length = sequence_len
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // numHeads
        assert self.head_dim * numHeads == emb_dim, "emb_dim must be divisable by numHeads"
        
        self.KQV_liner = nn.Linear(emb_dim , emb_dim * 3)
        self.out_liner = nn.Linear(emb_dim, emb_dim)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x : (B, sequence_len, emb_dim) 

        B = x.shape[0]
        x = self.KQV_liner(x)
        # x : (B, sequence_len, emb_len * 3)

        x = x.view(B, self.seqence_length, 3, self.numHeads, self.head_dim)
        # x : (B, sequence_len, 3"KQV", numHeads, head_dim)

        x = x.permute(2, 0, 3, 1, 4)
        # x : (3"KQV", B, numHeads, sequence_len, head_dim)

        K = x[0]
        Q = x[1]
        V = x[2]

        context = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        # context : (B, numHeads, squence_len, head_dim)
        context = context.transpose(1, 2).reshape(B, self.seqence_length, self.emb_dim)
        # context : (B, sequence_len, emb_dim)

        return self.out_liner(context)



