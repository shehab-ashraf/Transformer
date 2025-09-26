from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors
from datasets import DatasetDict

def build_tokenizer(dataset_dict: DatasetDict, vocab_size: int, min_frequency: int = 3, save_dir: str = 'checkpoints/tokenizer'):
    """
    Build and save a BPE tokenizer using all splits of a DatasetDict.
    
    Args:
        dataset_dict: DatasetDict containing train, validation, and test splits
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum frequency for tokens
        save_dir: Directory to save the tokenizer JSON file

    Returns:
        Tokenizer: The trained tokenizer object.
    """

    # 1. Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Normalizer
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(""", '"'),
        normalizers.Replace(""", '"'),
        normalizers.Replace("'", "'"),
        normalizers.Strip()
    ])

    # 3. Pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])

    # 4. Trainer
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]  # PAD is 0, UNK is 1, SOS is 2, EOS is 3
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # 5. Create data iterator for all splits
    def get_all_data():
        """Yields all texts from all splits for tokenizer training"""
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset_dict:
                split_data = dataset_dict[split_name]
                for item in split_data:
                    # Yield source text (German)
                    yield item['translation']['de']
                    # Yield target text (English)
                    yield item['translation']['en']

    # Train tokenizer on all data
    tokenizer.train_from_iterator(get_all_data(), trainer=trainer)

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

    return tokenizer