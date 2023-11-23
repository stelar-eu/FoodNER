from typing import List, Tuple

import torch

from config.nlp_models import (
    PT_MAX_SEQUENCE_LENGTH,
    PT_START_TOKEN,
    PT_END_TOKEN,
    PT_PAD_TOKEN
)
from src.NLP.datasets.pytorch import PytorchProcessor
from src.tools.general_tools import load_pickled_data


class PytorchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset class for the handling of the PyTorch datasets."""
    def __init__(
        self, 
        data_pickle_path: str,
        word2idx_path: str,
        label2idx_path: str
    ):
        self.bos_token = PT_START_TOKEN
        self.eos_token = PT_END_TOKEN
        self.pad_token = PT_PAD_TOKEN
        self.data = load_pickled_data(data_pickle_path)
        self.word2idx = load_pickled_data(word2idx_path)
        self.label2idx = load_pickled_data(label2idx_path)
        self.max_length = PT_MAX_SEQUENCE_LENGTH

    def __len__(self) -> int:
        """Return the length of the dataset. I.e. the number of sentences."""
        # In contrast to the spacy format we don't need to calculate the total
        # number of annotations, since, now, each sentence maps to a single
        # list of labels where each label correspond to each token.
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[List[str], List[str]]:
        """Return the idx-th item of the dataset."""
        return self.data[idx]


class Collator:
    def __init__(self, dataset: PytorchProcessor):
        self.dataset = dataset
        self.max_length: int = dataset.max_length
        self.bos_token: str = dataset.bos_token
        self.eos_token: str = dataset.eos_token
        self.pad_token: str = dataset.pad_token

    def __call__(self, batch):
        # TODO: George: Add the collate function
        # This should take in teh src (and tgt) sequences
        # and return a batch of tensors padded to the max length
        # and containing the start and end tokens.
        """Merges a list of (tokens, labels) to form a mini-batch.

        Args:
            batch: a list of (tokens, labels) where tokens is a list
                of token ids in a given sentence and labels is a list of
                label ids in the same sentence. Their lengths must be 
                equal.

        Returns:
            src_seqs of shape (MAX_LENGTH, batch_size): Tensor of padded
                source sequences.
            lab_seqs of shape (MAX_LENGTH, batch_size): Tensor of padded
                label sequences.
            lengths: List of lengths of each sequence in the batch.
        """
        lengths = []
        src_seqs = []
        lab_seqs = []
        for toks, lab in batch:
            lengths.append(len(toks))
            src_seqs.append(
                [self.bos_token]
                + toks
                + [self.eos_token]
                + [self.dataset.word2idx[self.pad_token]] * (self.max_length - len(toks))
            )
            lab_seqs.append(
                [self.bos_token]
                + lab
                + [self.eos_token]
                + [self.dataset.label2idx[self.pad_token]] * (self.max_length - len(lab))
            )
        src_seqs = torch.tensor(src_seqs).T
        lab_seqs = torch.tensor(lab_seqs).T

        return src_seqs, lab_seqs, lengths


def get_dataloader(
    data_pickle_path: str,
    word2idx_path: str,
    label2idx_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> torch.utils.data.DataLoader:
    dataset = PytorchDataset(
        data_pickle_path=data_pickle_path,
        word2idx_path=word2idx_path,
        label2idx_path=label2idx_path
    )
    collate_fn = Collator(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )