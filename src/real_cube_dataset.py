import torch
from torch.utils.data import Dataset
import os
from glob import glob
import natsort

class RealTossesDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = glob(os.path.join(data_dir, '*.pt'))
        
        self.file_paths = natsort.natsorted(self.file_paths)

        for path in self.file_paths:
            data_tensor = torch.load(path)
            data_tensor = data_tensor.squeeze(0)
            
            data_tensor = data_tensor.float()


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        data_tensor = torch.load(path)
        data_tensor = data_tensor.squeeze(0)
        
        data_tensor = data_tensor.float()

        # Format: pos(3), quat(4), vel(3), ang_vel(3), control(6)
        sample = {
            'position':    data_tensor[..., 0:3],
            'quaternion':  data_tensor[..., 3:7],
            'velocity':    data_tensor[..., 7:10],
            'ang_vel':     data_tensor[..., 10:13],
            'control':     data_tensor[..., 13:19],
            'full_state':  data_tensor
        }

        return sample

def pad_collate(batch):
    from torch.nn.utils.rnn import pad_sequence
    
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        items = [d[key] for d in batch]
        
        collated_batch[key] = pad_sequence(items, batch_first=True, padding_value=0.0)
        
    # Add a mask to know which steps are padding
    lengths = torch.tensor([d['full_state'].shape[0] for d in batch])
    collated_batch['lengths'] = lengths
    
    return collated_batch
