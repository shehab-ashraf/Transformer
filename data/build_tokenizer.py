from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors
from datasets import DatasetDict


def build_tokenizer(
    dataset_dict: DatasetDict, 
    vocab_size: int, 
    min_frequency: int = 3, 
) -> Tokenizer:
    """
    Build and train a BPE (Byte-Pair Encoding) tokenizer.
    
    This tokenizer is trained on both source (German) and target (English) texts from
    the dataset, creating a shared vocabulary for both languages.
    
    Args:
        dataset_dict: DatasetDict containing train, validation, and test splits with
                     'translation' field containing 'de' and 'en' keys
        vocab_size: Maximum size of the vocabulary
        min_frequency: Minimum frequency threshold for including tokens in vocabulary

    Returns:
        Tokenizer: Trained tokenizer ready for encoding/decoding text
    """
    

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # text normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),                    # Normalize to NFC 
        normalizers.Replace(""", '"'),        # Convert curly double quotes to straight quotes
        normalizers.Replace(""", '"'),
        normalizers.Replace("'", "'"),        # Convert curly single quote to straight apostrophe
        normalizers.Strip()                   # Remove leading/trailing whitespace
    ])

    # Split text on whitespace and separate punctuation before BPE training
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),          # Split on whitespace characters
        pre_tokenizers.Punctuation()          # Separate punctuation from words
    ])

    # Define special tokens with specific IDs:
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Function to yield text data
    def get_all_data():
        """
        Generator that yields all texts from all dataset splits.
        
        Iterates through train, validation, and test splits, yielding both
        German (source) and English (target) texts for tokenizer training.
        
        Yields:
            str: Individual text samples from the dataset
        """
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset_dict:
                split_data = dataset_dict[split_name]
                for item in split_data:
                    # Yield source language text (German)
                    yield item['translation']['de']
                    # Yield target language text (English)
                    yield item['translation']['en']

    # Train the BPE tokenizer on all available data
    tokenizer.train_from_iterator(get_all_data(), trainer=trainer)

    # Retrieve token IDs for special tokens
    SOS_token_id = tokenizer.token_to_id("[SOS]")
    EOS_token_id = tokenizer.token_to_id("[EOS]")

    # Configure post-processor to automatically add [SOS] and [EOS] tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SOS]:0 $A:0 [EOS]:0",       # $A represents the input text
        special_tokens=[
            ("[SOS]", SOS_token_id),
            ("[EOS]", EOS_token_id),
        ],
    )

    return tokenizer