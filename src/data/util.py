ENCODER_REGISTRY = {}


def register_encoder(name):
    def add_to_registry(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return add_to_registry


DATASET_REGISTRY = {}


def register_dataset(name):
    def add_to_registry(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return add_to_registry



def visualize_attention_matrix(tokens, attn_mat, ignore_pad=False):
    if ignore_pad is True:
        max_token_len = max([len(tok) for tok in tokens if tok != "[PAD]"])
    else:
        max_token_len = max([len(tok) for tok in tokens])
    header = f"{'':<{max_token_len+2}}"
    for tok in tokens:
        if ignore_pad is True and tok == "[PAD]":
            continue
        header += f"{tok:<{len(tok)+2}}"
    print(header)
    print('-' * len(header))

    for (i, row) in enumerate(attn_mat):
        if i > max_token_len:
            break
        row_str = f"{tokens[i]:<{max_token_len+1}}| "
        for (j, item) in enumerate(row):
            row_str += f"{item.item():<{len(tokens[j])+2}}"
        print(row_str)
