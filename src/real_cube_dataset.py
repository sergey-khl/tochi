import torch
from torch.utils.data import Dataset
import os
from glob import glob
import natsort
import numpy as np

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

    def z_up_to_y_up(self, state_tensor):
        """
        position (3), quaternion (4), velocity (3), angular velocity (3), control (6),
        convert z up tensor to y up 
        """
        new_state = state_tensor.clone()
        
        # (x, y, z) -> (x, z, -y)
        def rotate_vector_indices(new_state, start_idx):
            new_state[start_idx + 1], new_state[start_idx + 2] = new_state[start_idx + 2], -new_state[start_idx + 1]

        # --- Apply Vector Rotations ---
        # Position
        rotate_vector_indices(new_state, 0)
        # Velocity
        rotate_vector_indices(new_state, 7)
        # Angular Velocity
        rotate_vector_indices(new_state, 10)
        # control stuff, should be just 0 for this dataset but change anyways
        rotate_vector_indices(new_state, 13)
        rotate_vector_indices(new_state, 16)

        qw, qx, qy, qz = new_state[3], new_state[4], new_state[5], new_state[6]
        q_converted_x = qx
        q_converted_y = -qz
        q_converted_z = qy
        q_converted_w = qw
        rad = np.radians(-90) # find using right hand rule (curl da fingies)
        
        corr_x = np.sin(rad / 2)
        corr_y = 0.0
        corr_z = 0.0
        corr_w = np.cos(rad / 2)
        q_final_x = q_converted_w * corr_x + q_converted_x * corr_w + q_converted_y * corr_z - q_converted_z * corr_y
        q_final_y = q_converted_w * corr_y - q_converted_x * corr_z + q_converted_y * corr_w + q_converted_z * corr_x
        q_final_z = q_converted_w * corr_z + q_converted_x * corr_y - q_converted_y * corr_x + q_converted_z * corr_w
        q_final_w = q_converted_w * corr_w - q_converted_x * corr_x - q_converted_y * corr_y - q_converted_z * corr_z
        new_state[3] = q_final_x
        new_state[4] = q_final_y
        new_state[5] = q_final_z
        new_state[6] = q_final_w
        return new_state

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        data_tensor = torch.load(path)
        data_tensor = data_tensor.squeeze(0)
        
        data_tensor = data_tensor.float()

        for i in range(data_tensor.shape[0]):
            data_tensor[i] = self.z_up_to_y_up(data_tensor[i])

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
