import jax.numpy as jnp


def init_positions_grid(rng, shape):
    """
    Initialize the latent poses on a grid using JAX.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        num_dims (int): The number of dimensions for each point.

    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. Shape [num_signals, num_latents, ...].
    """
    num_signals, num_latents, num_dims = shape

    # Ensure num_latents is a power of num_dims
    assert abs(round(num_latents ** (1. / num_dims), 5) % 1) < 1e-5, 'num_latents must be a power of the number of position dimensions'

    # Calculate the number of latents per dimension
    num_latents_per_dim = int(round(num_latents ** (1. / num_dims)))

    # Create an n-dimensional mesh grid [-1 to 1] for each dimension
    grid_axes = jnp.linspace(-1 + 1 / num_latents_per_dim, 1 - 1 / num_latents_per_dim, num_latents_per_dim)
    grids = jnp.meshgrid(*[grid_axes for _ in range(num_dims)], indexing='ij')

    # Stack and reshape to create the positions matrix
    positions = jnp.stack(grids, axis=-1).reshape(-1, num_dims)

    # Repeat for the number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions


def init_appearances_ones(num_latents: int, num_signals: int, latent_dim: int):
    """
    Initialize the latent features as ones using JAX.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.
        latent_dim (int): The dimensionality of the latent code.

    Returns:
        z_features (jax.numpy.ndarray): The latent features for each signal. Shape [num_signals, num_latents, latent_dim].
    """
    z_features = jnp.ones((num_signals, num_latents, latent_dim))
    return z_features


def init_orientations_fixed(num_latents:int, num_signals: int, num_dims: int):
    """ Initialize the latent orientations as fixed.

    Args:
        num_latents (int): The number of latent points.
        num_signals (int): The number of signals.

    Returns:
        z_orientations (torch.Tensor): The latent orientations for each signal. Shape [num_signals, num_latents, ...].
    """
    orientations = jnp.zeros((num_signals, num_latents, num_dims))
    return orientations