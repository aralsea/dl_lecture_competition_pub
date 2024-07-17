class SelfAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
