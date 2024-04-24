import torch
from torchsparseattn import Fusedmax  # noqa


class SparsegenLin(torch.nn.Module):
    """
    Generic sparsegen-lin function as described in

    Laha, A., Chemmengath, S. A., Agrawal, P., Khapra, M.,
    Sankaranarayanan, K., & Ramaswamy, H. G. (2018).
    On Controllable Sparse Alternatives to Softmax.
    Advances in Neural Information Processing Systems, 31.
    https://proceedings.neurips.cc/paper/2018/hash/6a4d5952d4c018a1c1af9fa590a10dda-Abstract.html  # noqa

    Implementation modified from
    https://github.com/KrisKorrel/sparsemax-pytorch

    As lam --> 1, output approaches one-hot vector.
    As lam --> -inf, output approaches uniform.
    """

    def __init__(self, dim=None, lam=0.0):
        """
        Args:
            dim (int, optional): The dimension over which to apply
                                 the sparsegen function.
            lam (float): The lambda parameter. Default 0.0.
        """
        super().__init__()

        self.dim = -1 if dim is None else dim
        assert lam < 1
        self.lam = lam

    def __str__(self):
        return f"SparsegenLin(dim={self.dim}, lam={self.lam})"

    def __repr__(self):
        return f"SparsegenLin(dim={self.dim}, lam={self.lam})"

    def forward(self, inputs):
        """Forward function.
        Args:
            inputs (torch.Tensor): Input tensor. First dimension is batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate inputs by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)  # noqa

        # Sort inputs in descending order.
        # (NOTE: Can be replaced with linear time selection method:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        ks = torch.arange(start=1, end=number_of_logits + 1, step=1,
                          dtype=inputs.dtype).view(1, -1)
        ks = ks.expand_as(zs).to(zs.device)

        # Determine sparsity of projection
        bound = 1 - self.lam + ks * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * ks, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1 + self.lam) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        ps = (inputs - taus) / (1 - self.lam)
        self.output = torch.max(torch.zeros_like(inputs), ps)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        grad_sum = torch.sum(grad_output * nonzeros, dim=dim)
        grad_sum /= torch.sum(nonzeros, dim=dim)
        self.grad_inputs = nonzeros * (grad_output - grad_sum.expand_as(grad_output))  # noqa

        return self.grad_inputs
