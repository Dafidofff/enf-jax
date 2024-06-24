import numpy as np
from torch.utils.data import Dataset


class LatentDataset(Dataset):

    def __init__(self, p, a, window, labels, transforms=None):
        self.p = p
        self.a = a
        self.window = window
        self.labels = labels

        if transforms is not None:
            self.transforms = transforms

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        p, a, window, labels = self.p[idx], self.a[idx], self.window[idx], self.labels[idx]
        if hasattr(self, 'transforms'):
            for transform in self.transforms:
                p, a, window, labels = transform(p, a, window, labels)
        return p, a, window, labels


def perturb_positions(p, a, window, label, perturbation_scale=0.2):
    """Perturb the positions.

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        perturbation_scale (float, optional): The perturbation scale. Defaults to 0.1.

    Returns:
        np.ndarray: The perturbed positions.
    """
    p_perturbed = p + np.random.randn(*p.shape) * perturbation_scale
    return p_perturbed, a, window, label


def perturb_appearance(p, a, window, label, perturbation_scale=0.2):
    """Perturb the appearance latents.

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        perturbation_scale (float, optional): The perturbation scale. Defaults to 0.1.

    Returns:
        np.ndarray: The perturbed appearance latents.
    """
    a_perturbed = a + np.random.randn(*a.shape) * perturbation_scale
    return p, a_perturbed, window, label


def drop_latents(p, a, window, label, drop_rate=0.2):
    """Drop latents. Only mask out appearance

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        drop_rate (float, optional): The drop rate. Defaults to 0.1.

    Returns:
        np.ndarray: The dropped latents.
    """
    mask = np.random.rand(*a.shape[:-1]) > drop_rate
    a_dropped = a * mask[..., None]
    return p, a_dropped, window, label
