from pathlib import Path
import jax.numpy as jnp

from torch.utils.data import Dataset


class DFTWithID(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.data_path = self.root / 'dft/LJ_data_BE053DCPX'
        self.density_files = list(self.data_path.glob('rho_*'))

    def __getitem__(self, index):
        density_file = self.density_files[index]
        numb = density_file.stem.split('_')[-1].split('.npy')[0]
        energy_file = self.data_path / f'dF_drho_{numb}.npy'

        density = jnp.load(density_file)
        energy = jnp.load(energy_file)
        return density, energy, index

    def __len__(self):
        return len(self.density_files)