import numpy as np
from torch.utils.data import Dataset
import torch
# Import whatever extra thing you like!


class PointDataset(Dataset):
    """ A dataset that reads npy files. """

    def __init__(self, file_path):
        """ Read the file using np.load and store them. """
        self.file_path = file_path
        self.data = torch.tensor(np.load(self.file_path), dtype=torch.float)
        

    def __getitem__(self, i):
        """ Return the i'th point from the dataset. """
        return self.data[i,:]


    def __len__(self):
        """ Return the size of the dataset (number of points) """
        return self.data.shape[0]
