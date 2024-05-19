import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple
from jaxtyping import Float, Int

class InductionHeadsDataset(Dataset):
    def __init__(self, num_samples: Int, length: Int, vocab_size: Int) -> None:
        """
        num_samples: number of samples in the dataset
        length: number of tokens to remember (actual segment length should be 2*length)
        vocab_size: number of possible tokens to choose from
        """
        self.num_samples = num_samples
        self.length = length
        self.vocab_size = vocab_size
        self.data, self.gaps = induction_heads_dataset(num_samples, length, vocab_size)

    def __len__(self) -> Int:
        return self.num_samples
    
    def __getitem__(self, idx: Int) -> Tuple[Tensor, Tensor]:
        """
        dataset: (1 + length*3)
        random_gap: Int
        """
        return self.data[idx], self.gaps[idx]


def induction_heads_dataset(batch: Int, length: Int, vocab_size: Int) -> Tuple[Tensor, Tensor]:
    """
    Generate the induction heads dataset. Format is length number of randomly generated
    tokens, then with the same tokens repeated. 
    dataset: (batch, 1 + length*3)
    random_gap: (batch,)
    """
    dataset = torch.zeros(batch, 1 + length*3).long()

    random_sequences = torch.randint(0, vocab_size, (batch, length))
    dataset[:, 1:length+1] = random_sequences
    random_gap = torch.randint(1, length, (batch,))
    for i in range(batch):
        dataset[i, length+1+random_gap[i]:length*2+1+random_gap[i]] = random_sequences[i]
    return dataset, random_gap

if __name__ == "__main__":
    ## Test dataset function
    print(induction_heads_dataset(4, 12, 8192))

