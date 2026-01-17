import torch
from torch.utils.data import Dataset, TensorDataset
import os
from glob import glob
import natsort
import numpy as np
import random

class RealCubeDataset(Dataset):
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        self.file_paths = glob(os.path.join(data_dir, '*.pt'))
        
        self.file_paths = natsort.natsorted(self.file_paths)

        self.device = device


    def __len__(self):
        return len(self.file_paths)

    def z_up_to_y_up(self, state_tensor):
        """
        state_tensor: position (3), quaternion (4), velocity (3), angular velocity (3), control (6),

        we need to convert from contactnets left hand rule with z up to warp which is 
        right hand rule with y up. This means we need to invert y, switch y and z, and rotate along x by -90
        """
        new_state = state_tensor.clone()
        
        # (x, y, z) -> (x, z, -y)
        def y_z_flip(new_state, start_idx):
            y = new_state[:, start_idx + 1].clone()
            z = new_state[:, start_idx + 2].clone()
            new_state[:, start_idx + 1] = z
            new_state[:, start_idx + 2] = -y

        # Position
        y_z_flip(new_state, 0)
        # Velocity
        y_z_flip(new_state, 7)
        # Angular Velocity
        y_z_flip(new_state, 10)
        # control stuff, should be just 0 for this dataset but change anyways
        y_z_flip(new_state, 13)
        y_z_flip(new_state, 16)

        # in contactnets they use quat convention of (w, x, y, z)
        qw, qx, qy, qz = new_state[:, 3].clone(), new_state[:, 4].clone(), new_state[:, 5].clone(), new_state[:, 6].clone()
        # same as before, flip y and z and invert y
        q_converted_x = qx
        q_converted_y = -qz
        q_converted_z = qy
        q_converted_w = qw

        rad = np.radians(-90) # find using right hand rule (curl da fingies)
        corr_x = np.sin(rad / 2)
        corr_y = 0.0
        corr_z = 0.0
        corr_w = np.cos(rad / 2)
        # quaternion rotation magic formula
        new_state[:, 3] = q_converted_w * corr_x + q_converted_x * corr_w + q_converted_y * corr_z - q_converted_z * corr_y
        new_state[:, 4] = q_converted_w * corr_y - q_converted_x * corr_z + q_converted_y * corr_w + q_converted_z * corr_x
        new_state[:, 5] = q_converted_w * corr_z + q_converted_x * corr_y - q_converted_y * corr_x + q_converted_z * corr_w
        new_state[:, 6] = q_converted_w * corr_w - q_converted_x * corr_x - q_converted_y * corr_y - q_converted_z * corr_z

        return new_state

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        data_tensor = torch.load(path)
        data_tensor = data_tensor.squeeze(0)
        
        data_tensor = data_tensor.float()

        data_tensor = self.z_up_to_y_up(data_tensor)

        sample = {
            'position':    data_tensor[..., 0:3],
            'quaternion':  data_tensor[..., 3:7],
            'velocity':    data_tensor[..., 7:10],
            'ang_vel':     data_tensor[..., 10:13],
            'control':     data_tensor[..., 13:19],
            'full_state':  data_tensor
        }

        return sample
    
    def load(self, splits=[50,30,20], num_tosses=100, noise=0.4):
        """
        created paired datasets
        the functions above are only used for previwing and this is for training.
        Probably could make a more uniform implementation
        but I want the same batches as cnets for easier debug

        Args:
            noise: how much Gaussian noise to add.
        """
        def process_run(data):
            """ Turn time sequenced data into a batch of paired time steps.

            Args:
                data: step_n x (state_n + control_n) * entity_n.

            Returns:
                step_n - 1 x 2 x (state_n + control_n) * entity_n
            """
            n = data.shape[0]
            batch = torch.zeros((n - 1, 2, data.shape[1]))

            for i in range(n - 1):
                batch[i, :, :] = torch.cat((data[i, :].unsqueeze(0),
                                            data[i + 1, :].unsqueeze(0)))
            return batch.to(self.device)

        def load_runs(idxs):
            datas = []

            try:
                for idx in idxs:
                    path = self.file_paths[idx]
                    data = torch.load(path).to(self.device)
                    data = data.squeeze(0).float()

                    data = self.z_up_to_y_up(data)

                    data = data + torch.randn(data.shape, device=self.device) * noise
                    datas.append(data)
            except Exception as err:
                print(f'Could not load data')
                print(err)

            return datas

        runs = list(range(self.__len__()))
        random.shuffle(runs)
        if num_tosses is not None:
            runs = runs[:num_tosses]

        split_points = np.cumsum(splits)
        split_points = [int(np.floor(split * 0.01 * len(runs))) for split in split_points]

        train_data = runs[0:split_points[0]]
        valid_data = runs[split_points[0]:split_points[1]]
        test_data = runs[split_points[1]:]

        self.train_runs = load_runs(train_data)
        self.valid_runs = load_runs(valid_data)
        self.test_runs  = load_runs(test_data)


        train_processed = list(map(process_run, self.train_runs))
        valid_processed = list(map(process_run, self.valid_runs))
        test_processed  = list(map(process_run, self.test_runs))

        self.train = TensorDataset(torch.cat(train_processed))
        self.valid = TensorDataset(torch.cat(valid_processed))
        self.test = TensorDataset(torch.cat(test_processed))

