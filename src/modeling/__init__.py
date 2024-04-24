from .util import MODEL_REGISTRY, TOKEN_POOLER_REGISTRY

# Populates MODEL_REGISTRY
from .default import StanceModel
from .default_t5_gen import StanceModelT5Gen
from .pooling import StancePoolingModel
from .attention_pooling import StancePoolingModelWithAttention

# Populates TOKEN_POOLER_REGISTRY
from .token_poolers import (MaxEntityPooler,
                            MeanEntityPooler,
                            SoftmaxAttentionEntityPooler,
                            SparsegenAttentionEntityPooler)
