def get_lm_probs(text, lm):
    # lm = MT5ForConditionalGeneration
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    dids = lm._shift_right(input_ids)
    lm_probs = []
    for (i, did) in enumerate(dids[0]):
        out = lm(input_ids=input_ids[:, :i+1], decoder_input_ids=dids[:, :i+1])
        next_tid = dids[:, i].squeeze().item()
        lm_probs.append(out.logits.squeeze(0).softmax(-1)[-1][next_tid].item())
    return torch.as_tensor(lm_probs)
