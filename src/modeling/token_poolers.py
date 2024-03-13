import torch
from torchsparseattn import Fusedmax

from .util import register_token_pooler


class BaseEntityPooler(torch.nn.Module):

    def __init__(self, insize, outsize):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.insize, self.outsize),
            torch.nn.Tanh())

    def forward(self, hidden, token_mask):
        # token_mask.shape == hidden.shape (batch, max_length, hidden_dim)
        #  is a binary float vector with 1s in the positions to keep
        #  and 0s everywhere else.
        assert token_mask.size() == hidden.size()
        masked_hidden = hidden * token_mask
        pooled = self.pool_fn(masked_hidden, token_mask)
        # Project it down, if the outsize is different.
        if self.outsize == self.insize:
            projected = pooled
        else:
            projected = self.output_layer(pooled)
        return projected

    def string(self):
        return f"{self.__class__.__name__}({self.insize}, {self.outsize})"

    def pool_fn(self, masked_hidden, token_mask):
        raise NotImplementedError()


@register_token_pooler("mean")
class MeanEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        # Replace masked with nan and then compute using torch.nanmean
        masked_hidden[torch.logical_not(token_mask)] = torch.nan
        pooled = torch.nanmean(masked_hidden, axis=1)
        if torch.isnan(pooled).any():
            raise ValueError("No masked values found in MeanEntityPooler")
        pooled = torch.nan_to_num(pooled)
        return pooled


@register_token_pooler("max")
class MaxEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        # Replace masked with -inf to avoid zeroing out
        # hidden dimensions if the non-masked values are all negative.
        masked_hidden[torch.logical_not(token_mask)] = -torch.inf
        pooled = torch.max(masked_hidden, axis=1)[0]
        if torch.isinf(pooled).all():
            raise ValueError("No masked values found in MaxEntityPooler")
        pooled = torch.nan_to_num(pooled, neginf=-1.0)
        return pooled


@register_token_pooler("first")
class FirstEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        first_nonzero_idxs = token_mask.max(1).indices[:, 0]
        batch_idxs = torch.arange(token_mask.size(0))
        pooled = masked_hidden[batch_idxs, first_nonzero_idxs, :]
        return pooled


@register_token_pooler("last")
class LastEntityPooler(BaseEntityPooler):

    def pool_fn(self, masked_hidden, token_mask):
        pass


class BaseAttentionEntityPooler(BaseEntityPooler):

    def __init__(self, insize, outsize, **projection_fn_kwargs):
        super().__init__(insize, outsize)
        # multiply outsize by 2 because we concatenate target and
        # body representations as input to the alignment_model.
        self.alignment_model = torch.nn.Linear(2 * outsize, 1)
        self.projection_fn_kwargs = projection_fn_kwargs

    def forward(self, hidden, token_mask, pooled_tokens):
        """
        Compute attention between the hidden representations in the token_mask
        and the pooled_tokens.
        hidden: [batch, max_length, hidden_dim]
        token_mask: [batch, max_length, hidden_dim]
        pooled_tokens: [batch, hidden_dim]
        """
        assert token_mask.size() == hidden.size()
        masked_hidden = hidden * token_mask
        pooled, attentions = self.pool_fn(
            masked_hidden, token_mask, pooled_tokens)
        # Project it down, if the outsize is different.
        if self.outsize == self.insize:
            projected = pooled
        else:
            projected = self.output_layer(pooled)
        return projected, attentions

    def pool_fn(self, masked_hidden, token_mask, pooled_tokens):
        """
        projection_fn = some.projection_fn
        return self.generic_attention_pooler(
            masked_hidden, token_mask, pooled_tokens, projection_fn)
        OR
        return self.generic_cross_attention_pooler()
        """
        raise NotImplementedError()

    def generic_attention_pooler(self, masked_hidden, token_mask,
                                 pooled_tokens, projection_fn):
        """
        projection_fn: A function that maps scores to the simplex,
            e.g. softmax.
        """
        tokens_repeated = pooled_tokens.unsqueeze(1).repeat(
            1, masked_hidden.size(1), 1)
        tokens_repeated_masked = tokens_repeated * token_mask
        # Compute alignments between the pooled_tokens and each
        # hidden token representation.
        alignment_inputs = torch.cat((tokens_repeated_masked, masked_hidden),
                                     dim=2)
        attention_scores = self.alignment_model(alignment_inputs)
        attention_mask = token_mask[:, :, 0].bool()
        # Project the attention scores for each example.
        # We need to do it example by example because each has a different
        # token mask.
        attention_probs = torch.zeros_like(attention_scores)
        batch_size = masked_hidden.size(0)
        for example_idx in range(batch_size):
            masked_scores = torch.masked_select(
                attention_scores[example_idx],
                attention_mask[example_idx].unsqueeze(1))
            if masked_scores.nelement() == 0:
                probs = torch.empty_like(masked_scores)
            else:
                probs = projection_fn(masked_scores)
            prob_idxs = attention_mask[example_idx]
            attention_probs[example_idx][prob_idxs] = probs.unsqueeze(1)
        # Scale the token representations by their attention probabilities
        # and sum over the token dimension to obtain the weighted average.
        pooled = (masked_hidden * attention_probs).sum(1)
        return pooled, attention_probs
                                        

    def generic_cross_attention_pooler(self, masked_hidden, token_mask,
                                       pooled_tokens, projection_fn):
        # Implement CrossAttention class here.
        # return skip_connection_outputs, attention_weights
        pass


@register_token_pooler("attention-softmax")
class SoftmaxAttentionEntityPooler(BaseAttentionEntityPooler):

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = torch.nn.Softmax(dim=0)
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


@register_token_pooler("attention-sparsegen")
class SparsegenAttentionEntityPooler(BaseAttentionEntityPooler):

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = SparsegenLin(dim=0, **self.projection_fn_kwargs)
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


@register_token_pooler("attention-rela")
class ReLAEntityPooler(BaseAttentionEntityPooler):
    """
    Rectified Linear Attention from

    Zhang, B., Titov, I., & Sennrich, R. (2021).
    Sparse Attention with Linear Units (arXiv:2104.07012). arXiv.
    https://doi.org/10.48550/arXiv.2104.07012
    """
    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = torch.nn.ReLU()
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


@register_token_pooler("attention-fusedmax")
class FusedmaxAttentionEntityPooler(BaseAttentionEntityPooler):

    def pool_fn(self, masked_hidden, entity_mask, pooled_entities):
        projection_fn = Fusedmax(**self.projection_fn_kwargs)
        return self.generic_attention_pooler(
            masked_hidden, entity_mask, pooled_entities, projection_fn)


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
