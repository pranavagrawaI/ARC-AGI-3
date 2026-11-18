def prepare_input(tokenizer):
    """Keep the native engine request independent of tokenizer return defaults."""
    messages = [
        {
            "role": "user",
            "content": "Find the smallest positive integer divisible by every integer from 1 through 20. Explain your reasoning and give the integer as your final answer.",
        }
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )
    if (
        not isinstance(ids, list)
        or not ids
        or not all(type(token) is int for token in ids)
    ):
        raise TypeError("Expected a nonempty flat list of integer input IDs")
    return messages, ids
