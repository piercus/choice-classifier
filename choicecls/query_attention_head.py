import torch
from torch import nn
import math
# classes

class QueryAttentionHead(nn.Module):
    def __init__(
        self,
        dim_key,
        num_choices,
        dim_query,
        inner_dim=64
    ):
        super().__init__()
        self.to_k = nn.Linear(dim_key, inner_dim, bias = False)
        self.to_q = nn.Linear(dim_query, inner_dim, bias = False)
        self.out = nn.Softmax(dim=1)
        self.num_choices = num_choices

    def forward(
        self,
        keys,
        query
    ):
        k, q = (self.to_k(keys), self.to_q(query))
        batch_size = k.shape[0]
        v = torch.eye(self.num_choices, dtype=torch.float32).unsqueeze(0).expand([batch_size, -1, -1]).to(k.get_device())
        q = q.unsqueeze(1)
        # print('v', v.shape)
        # print('q', q.shape)
        # print('k', k.shape)
        
        # inspired from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html?highlight=attention#torch.nn.functional.scaled_dot_product_attention
        out = torch.matmul(
            torch.softmax(
                torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)),
                dim=-1
            ),
        v).squeeze(1)
        # print('out', out.shape)
        return out
