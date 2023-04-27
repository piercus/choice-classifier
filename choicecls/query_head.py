import torch
from torch import nn

# classes

class QueryHead(nn.Module):
    def __init__(
        self,
        dim_query,
        dim_context,
        inner_dim=64
    ):
        super().__init__()
        self.to_q = nn.Linear(dim_query, inner_dim, bias = False)
        self.to_c = nn.Linear(dim_context, inner_dim, bias = False)
        self.out = nn.Softmax(dim=1)

    def forward(
        self,
        query,
        context
    ):
        q, c = (self.to_q(query), self.to_c(context))
        return self.out(torch.matmul(q, c.unsqueeze(2)).squeeze(2))
