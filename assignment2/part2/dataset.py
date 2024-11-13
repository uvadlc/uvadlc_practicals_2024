from argparse import Namespace

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, datafile_path: str, return_tensors: bool = False):
        data = open(datafile_path, 'r').read()
        chars = sorted(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.vocab_size = vocab_size
        
        self.string_to_index = { ch:i for i,ch in enumerate(chars) }
        self.index_to_string = { i:ch for i,ch in enumerate(chars) }
        self.return_tensors = return_tensors

    def encode(self, string: str) -> torch.Tensor:
        if self.return_tensors:
            out = torch.tensor([self.string_to_index[ch] for ch in string], dtype=torch.long)
        else:
            out = [self.string_to_index[ch] for ch in string]
        return out

    def decode(self, indices: torch.Tensor | list[int]) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.index_to_string[i] for i in indices])

class TextDataset(Dataset):
    """
    A dataset class for character-level text data processing.

    This class is designed to emit batches of characters from a given text file. It initializes 
    with the text data, creating mappings from characters to indices and vice versa, and calculates 
    the size of the dataset and vocabulary. It also provides methods to get the size of the 
    vocabulary, the block size for data chunks, and to retrieve a specific item from the dataset.

    Attributes:
        config (Namespace): Configuration object containing settings for the dataset.
        data (str): The entire text data loaded from the datafile.
        string_to_index (dict): A dictionary mapping each character to a unique integer.
        index_to_string (dict): A dictionary mapping each integer back to its corresponding character.
        block_size (int): The size of each data block (chunk of characters) to be returned.
        vocabulary_size (int): The number of unique characters in the dataset.

    Methods:
        get_vocab_size(): Returns the size of the vocabulary.
        get_block_size(): Returns the block size of the dataset.
        __len__(): Returns the number of blocks available in the dataset.
        __getitem__(idx): Returns a tuple of tensors (x, y) for training, where x is the input tensor 
                          and y is the target tensor, both derived from the dataset at the specified index.

    Parameters:
        config (Namespace): Configuration object for the dataset.
        datafile_path (str): Path to the text file containing the dataset.
        block_size (int, optional): Size of each data block. Defaults to 128.

    Raises:
        IOError: If the datafile_path does not lead to a valid file.

    Example:
        >>> dataset = TextDataset(config, 'path/to/textfile.txt', 128)
        >>> print(dataset[0])  # Get the first data block
    """

    def __init__(self, config: Namespace, datafile_path: str, block_size:int = 128, tokenizer: CharTokenizer = None):
        self.config = config

        # Load text data
        data = open(datafile_path, 'r').read() 
        self.data = data

        self.block_size = block_size

        self.tokenizer = tokenizer(datafile_path)
        
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = self.tokenizer.encode(chunk)
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y