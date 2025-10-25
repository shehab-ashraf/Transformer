"""
Builds a Byte-Pair Encoding (BPE) tokenizer for the translation task.
The tokenizer learns a vocabulary of subword units from the data.

What is BPE?
------------
BPE (Byte-Pair Encoding) is a subword tokenization algorithm:
• Starts with characters
• Iteratively merges frequent character pairs
• Creates vocabulary of subword units

Special Tokens:
---------------
• [PAD]: Padding token (makes sequences same length)
• [UNK]: Unknown token (rarely used with BPE)
• [SOS]: Start of sequence token
• [EOS]: End of sequence token

Shared Vocabulary:
i trained ONE tokenizer on BOTH German and English.
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors
from datasets import DatasetDict


def build_tokenizer(
    dataset_dict: DatasetDict, 
    vocab_size: int, 
    min_frequency: int = 3,
) -> Tokenizer:
    """
    Build and train a BPE tokenizer on the dataset.
    Creates a shared vocabulary for both source and target languages by
    training on all available text data (train + validation + test).
    
    Args:
        dataset_dict (DatasetDict): HuggingFace dataset
        vocab_size (int): Maximum vocabulary size to learn
        min_frequency (int): Minimum times a token must appear to be included
    
    Returns:
        Tokenizer: Trained BPE tokenizer ready for encoding/decoding
                  - Automatically adds [SOS] and [EOS] tokens
    """
    
    # Create base BPE tokenizer with unknown token
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))


    # Normalizers clean up text before tokenization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace(""", '"'),
        normalizers.Replace(""", '"'),
        normalizers.Replace("'", "'"),
        normalizers.Strip()
    ])

    # Pre-tokenizers split text into initial pieces before BPE merging
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation()
    ])

    
    # Special tokens
    special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    
    # Trainer handles the actual vocabulary learning
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,           
        min_frequency=min_frequency,     
        special_tokens=special_tokens,   
        show_progress=True,              
    )
    
    # Generator that yields all text from the dataset.
    def get_all_data():
        """
        Iterates through all splits (train, validation, test) and yields
        both source (German) and target (English) sentences.
        
        Using a generator is memory-efficient: doesn't load all text at once.
        """
        # Iterate through all available splits
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset_dict:
                split_data = dataset_dict[split_name]
                
                # Yield both source and target language texts
                for item in split_data:
                    # Yield German text (source)
                    yield item['translation']['de']
                    # Yield English text (target)
                    yield item['translation']['en']
    
    # Learn BPE merges from dataset
    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(get_all_data(), trainer=trainer)
    print(f"✓ Tokenizer trained with vocab size: {tokenizer.get_vocab_size()}")
    
    # Get token IDs for special tokens
    SOS_token_id = tokenizer.token_to_id("[SOS]")
    EOS_token_id = tokenizer.token_to_id("[EOS]")

    # Post-processor automatically wraps encoded sequences with [SOS] and [EOS]
    # Template: [SOS] <input_text> [EOS]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[SOS]:0 $A:0 [EOS]:0",
        special_tokens=[
            ("[SOS]", SOS_token_id),
            ("[EOS]", EOS_token_id),
        ],
    )

    return tokenizer