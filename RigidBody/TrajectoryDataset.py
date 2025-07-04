from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    """TRAJECTORY DATASET"""

    def __init__(self, dataframe, model = "RB"):
        if model == "RB":
            self.features = np.vstack((dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"])).transpose()
            self.targets  = np.vstack((dataframe["mx"], dataframe["my"], dataframe["mz"])).transpose()
            self.mid= 0.5*(self.features+self.targets)
        elif model == "HT":
            self.features = np.vstack((dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"], dataframe["old_rx"], dataframe["old_ry"], dataframe["old_rz"])).transpose()
            self.targets  = np.vstack((dataframe["mx"], dataframe["my"], dataframe["mz"], dataframe["rx"], dataframe["ry"], dataframe["rz"])).transpose()
            self.mid= 0.5*(self.features+self.targets)
        elif model in ["P3D", "K3D"]:
            self.features = np.vstack((dataframe["old_rx"], dataframe["old_ry"], dataframe["old_rz"], dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"])).transpose()
            self.targets  = np.vstack((dataframe["rx"], dataframe["ry"], dataframe["rz"], dataframe["mx"], dataframe["my"], dataframe["mz"])).transpose()
            self.mid= 0.5*(self.features+self.targets)
        elif model == "P2D":
            self.features = np.vstack((dataframe["old_rx"], dataframe["old_ry"], dataframe["old_mx"], dataframe["old_my"])).transpose()
            self.targets  = np.vstack((dataframe["rx"], dataframe["ry"], dataframe["mx"], dataframe["my"])).transpose()
            self.mid= 0.5*(self.features+self.targets)
        elif model == "Sh":
            self.features = np.vstack((dataframe["old_u"], dataframe["old_x"], dataframe["old_y"], dataframe["old_z"])).transpose()
            self.targets  = np.vstack((dataframe["u"], dataframe["x"], dataframe["y"], dataframe["z"])).transpose()
            self.mid= 0.5*(self.features+self.targets)
        else:
            raise Exception("Unknown model.")
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx, :], self.targets[idx, :], self.mid[idx, :])
