import torch


class CrossAttention(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        WQ = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.register_parameter("WQ", torch.nn.init.xavier_uniform_(WQ))
        WK = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.register_parameter("WK", torch.nn.init.xavier_uniform_(WK))
        WV = torch.nn.Parameter(torch.empty(self.input_dim, self.hidden_dim))
        self.register_parameter("WV", torch.nn.init.xavier_uniform_(WV))
        Wout = torch.nn.Parameter(torch.empty(self.hidden_dim, self.input_dim))
        self.register_parameter("Wout", torch.nn.init.xavier_uniform_(Wout))
        self.projection_fn = torch.nn.Softmax(dim=1)

    def forward(self, h_1, h_2):
        """
        h_1 \\in R^{T, D}
        h_2 \\in R^{S, D}
        """
        Q = torch.matmul(h_2, self.WQ)
        K = torch.matmul(h_1, self.WK)
        V = torch.matmul(h_1, self.WV)
        scores = torch.matmul(Q, K.T)
        weights = self.projection_fn(scores)
        avg = torch.matmul(weights, V)
        skip = h_2 + torch.matmul(avg, self.Wout)
        return skip, weights


class AdditiveAttention(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sim = torch.nn.Linear(self.input_dim*2, 1)
        self.projection_fn = torch.nn.Softmax(dim=1)

    def forward(self, h_1, h_2):
        eta_1 = h_1.mean(0).repeat(h_2.size(0), 1)
        catted = torch.hstack([eta_1, h_2])
        scores = self.sim(catted)
        weights = self.projection_fn(scores)
        avg = (weights * h_2)
        return avg, weights


def test():
    T = 4
    S = 3
    H = 5
    D = 2
    print(f"T: {T}")
    print(f"S: {S}")
    print(f"H: {H}")
    print(f"D: {D}")
    print()
    ca = CrossAttention(H, D)
    aa = AdditiveAttention(H, D)

    h_1 = torch.randn(T, H)
    h_2 = torch.randn(S, H)
    outputs, attn_weights = ca(h_1, h_2)
    print("Attention weights b/n S and T")
    print("  ", attn_weights.size())
    print("Output representations per s in S")
    print("  ", outputs.size())
    print()

    outputs, attn_weights = aa(h_1, h_2)
    print("Attention weights b/s S and average T")
    print("  ", attn_weights.size())
    print("Output representations per s in S")
    print("  ", outputs.size())


if __name__ == "__main__":
    test()
