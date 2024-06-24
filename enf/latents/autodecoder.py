import jax.numpy as jnp
from flax import linen as nn

from enf.latents.utils import init_positions_grid


class PositionOrientationFeatureAutodecoder(nn.Module):
    """
    Autodecoder module for position, orientation, and context decoding.

    Args:
        num_signals (int): Number of signals.
        num_latents (int): Number of latent variables.
        latent_dim (int): Dimensionality of the latent space.
        num_pos_dims (int): Number of position dimensions.
        num_ori_dims (int): Number of orientation dimensions.
        gaussian_window_size (float, optional): Size of the Gaussian window. Defaults to None.
        frequency_parameter (float, optional): Frequency parameter. Defaults to None.
    """

    num_signals: int
    num_latents: int
    latent_dim: int
    num_pos_dims: int
    num_ori_dims: int
    gaussian_window_size: float = None
    frequency_parameter: float = None

    def setup(self):
        """
        Setup method to initialize the model parameters.

        Initializes the latent positions, orientations, context, Gaussian window, and frequency parameters.
        """
        # Initialize the latent positions, orientations, appearances, gaussian window, and frequency parameter here
        # Note: Flax manages parameters differently, so we use self.param to declare model parameters
        self.p_pos = self.param('p_pos', init_positions_grid, (self.num_signals, self.num_latents, self.num_pos_dims))
        if self.num_ori_dims > 0:
            self.p_ori = self.param('p_ori', nn.initializers.zeros, (self.num_signals, self.num_latents, self.num_ori_dims))
        else:
            self.p_ori = None

        self.c = self.param('c', nn.initializers.ones, (self.num_signals, self.num_latents, self.latent_dim))

        if self.gaussian_window_size is not None:
            # Calculate gaussian window size s.t. each gaussian overlaps.
            # This is the same as setting the standard deviation to the distance between the latent points.
            # Create a grid of latent positions
            assert round(self.num_latents ** (1. / self.num_pos_dims), 5) % 1 == 0, 'num_latents must be a power of the number of position dimensions'

            # Create position orientation grid for the latent points
            num_latents_per_dim = int(round(self.num_latents ** (1. / self.num_pos_dims), 5))

            # Since our domain ranges from -1 to 1, the distance between each latent point is 2 / num_latents_per_dim
            # We want each gaussian to be centered at a latent point, and be removed 2 std from other latent points.
            gaussian_window_size = self.num_pos_dims / num_latents_per_dim

            self.gaussian_window = self.param('gaussian_window', nn.initializers.constant(gaussian_window_size), (self.num_signals, self.num_latents, 1))

    def __call__(self, idx: int):
        """
        Forward pass method.

        Args:
            idx (int): Index of the input.

        Returns:
            Tuple: Tuple containing the latent positions, appearances, and Gaussian window (if applicable).
        """
        # Gather the poses p_i = [p_pos_i, p_ori_i]
        p_pos = self.p_pos[idx]
        if self.num_ori_dims > 0:
            p_ori = self.p_ori[idx]
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        # Get the pose specific context c_i
        a = self.c[idx]

        # Optionally, get the gaussian window for the latent points
        if self.gaussian_window_size is not None:
            gaussian_window = self.gaussian_window[idx]
        else:
            gaussian_window = None
        return p, a, gaussian_window
    


class PositionOrientationFeatureAutodecoderMeta(PositionOrientationFeatureAutodecoder):
    """
    Autodecoder module for position, orientation, and feature decoding. This class differs 
    from the PositionOrientationFeatureAutodecoder class in that it does not keep track of sample
    specific parameters, however it keeps track of the meta-learned initialisation parameters.

    Args:
        num_signals (int): Number of signals.
        num_latents (int): Number of latent variables.
        latent_dim (int): Dimensionality of the latent space.
        num_pos_dims (int): Number of position dimensions.
        num_ori_dims (int): Number of orientation dimensions.
        gaussian_window_size (float, optional): Size of the Gaussian window. Defaults to None.
        frequency_parameter (float, optional): Frequency parameter. Defaults to None.
    """

    def __call__(self):
        # Gather the poses p_i = [p_pos_i, p_ori_i]
        p_pos = self.p_pos
        if self.num_ori_dims > 0:
            p_ori = self.p_ori
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        # Get the pose specific context c_i
        c = self.c

        # Optionally, get the gaussian window for the latent points
        if self.gaussian_window_size is not None:
            gaussian_window = self.gaussian_window
        else:
            gaussian_window = None
        return p, c, gaussian_window
