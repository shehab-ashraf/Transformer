from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors

def get_all_data(data):
    """
    Yields individual texts for tokenizer training.
    
    Args:
        data: Dataset containing translation pairs
        
    Yields:
        Individual texts for tokenizer training, alternating between source and target
    """
    for item in data:
        # Yield source text
        yield item['translation_src']
        # Yield target text
        yield item['translation_trg']

def build_tokenizer(data, vocab_size, min_frequency=5, save_dir='checkpoints/tokenizer'):
    """
    Build and save a BPE tokenizer for a given language from IWSLT2017 dataset.

    Args:
        save_dir (str): Directory to save the tokenizer JSON file.
        min_frequency (int): Minimum frequency of subwords to be kept.
        vocab_size (int or None): Maximum vocabulary size (None = unlimited).

    Returns:
        Tokenizer: The trained tokenizer object.
    """

    # 1. Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Normalizer
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("“", '"'),
        normalizers.Replace("”", '"'),
        normalizers.Replace("’", "'"),
        normalizers.Strip()
    ])

    # 3. Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])

    # 4. Trainer
    special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    trainer = trainers.BpeTrainer(
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        vocab_size=vocab_size
    )

    # 5. Train tokenizer
    tokenizer.train_from_iterator(get_all_data(data), trainer=trainer)

    # Get token IDs
    SOS_token_id = tokenizer.token_to_id("[SOS]")
    EOS_token_id = tokenizer.token_to_id("[EOS]")

    # Add post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SOS]:0 $A:0 [EOS]:0",
        special_tokens=[
            ("[SOS]", SOS_token_id),
            ("[EOS]", EOS_token_id),
        ],
    )

    # 6. Save
    save_path = f"{save_dir}/tokenizer.json"
    tokenizer.save(save_path)

    return tokenizer