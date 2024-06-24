import jax
import jax.numpy as jnp
import numpy as np
import trimesh

from experiments.fitting.trainers.shape.utils.common import make_3d_grid
from experiments.fitting.trainers.shape.utils.libmcubes.mcubes import marching_cubes
from experiments.fitting.trainers.shape.utils.libmise import MISE


def extract_mesh_from_neural_field(
    apply_model_fn,
    points_batch_size: int = 32000,
    threshold: float = 0,
    resolution0: int = 32,
    upsampling_steps: int = 3,
    padding: float = 0.1,
):
    """Mesh Extraction class for occupancy neural fields.

    Adapted from https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/onet/generation.py.

    Args:
        trainer: trainer object.
        shape_idx (int): ID of the model to generate mesh for.
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value, determines which points are considered in/outside of mesh
        resolution0 (int): start resolution for MISE
        upsampling_steps (int): number of upsampling steps
        padding (float): how much padding should be used for MISE
        simplify_mesh (bool): whether to simplify the mesh

    Returns:
        mesh (trimesh.Trimesh): generated mesh

    Note: uses MISE to obtain a high-resolution voxel grid, then uses marching cubes to extract the mesh.
    It's possible to not use MISE and directly use marching cubes on a low-resolution voxel grid, by
    setting upsampling_steps=0 and resolution0 to the desired resolution of the voxel grid.
    """

    # Compute bounding box size
    box_size = 2 + padding

    # Shortcut
    if upsampling_steps == 0:
        nx = resolution0
        pointsf = (1 + padding) * make_3d_grid(min_val=-1.0, max_val=1.0, resolution=resolution0)
        values = batch_eval_points(
            apply_model_fn=apply_model_fn,
            points=pointsf,
            points_batch_size=points_batch_size,
        )
        voxel_grid = - values.reshape(nx, nx, nx)
    else:
        mesh_extractor = MISE(
            resolution_0=resolution0,
            depth=upsampling_steps,
            threshold=threshold,
        )

        points = mesh_extractor.query()

        while points.shape[0] != 0:
            # Query points
            pointsf = jnp.array(points)
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = box_size * (pointsf - 0.5)
            # Evaluate model and update
            values = batch_eval_points(
                apply_model_fn=apply_model_fn,
                points=pointsf,
                points_batch_size=points_batch_size,
            )
            # NOTE: since we use SDF, we need to pass negative values to MISE
            values = - values.astype(np.float64)
            try:
                mesh_extractor.update(points, values.squeeze())
            except ValueError:
                print("Error in mesh extractor")
                break
            points = mesh_extractor.query()

        voxel_grid = mesh_extractor.to_dense()

    mesh = extract_mesh_from_voxel_grid(voxel_grid, threshold=threshold)
    return mesh


def extract_mesh_from_voxel_grid(occ_hat, padding=0.1, threshold=0.0):
    """Extracts the mesh from the predicted occupancy grid.

    Args:
        occ_hat (tensor): voxel grid of occupancies
        padding (float): how much padding should be used in defining the voxel grid
        threshold (float): threshold value, determines which points are considered in/outside of mesh
    """
    # Some short hands
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + padding

    # Make sure that mesh is watertight
    occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)
    vertices, triangles = marching_cubes(occ_hat_padded, threshold)

    # Strange behaviour in libmcubes: vertices are shifted by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=None, process=False)

    # Directly return if mesh is empty
    if vertices.shape[0] == 0:
        return mesh

    return mesh


def batch_eval_points(
    apply_model_fn, points: jax.Array, points_batch_size: int = 10000
) -> np.ndarray:
    """Evaluates the occupancy values for the points.

    Args:
        trainer: trainer object.
        points (jax.Array): points
        shape_idx (int): ID of the model to generate mesh for.
        points_batch_size (int): batch size for points evaluation
    """
    # p_split = torch.split(p, self.points_batch_size)
    sdf_hats = []

    # Unsqueeze points
    points = jnp.expand_dims(points, axis=0)

    # Loop over points in batches of size self.points_batch_size
    for start in range(0, points.shape[1], points_batch_size):
        p_split = points[:, start : start + points_batch_size]
        sdf_hat = apply_model_fn(x=p_split)
        sdf_hats.append(sdf_hat)

    # Concatenate and convert to numpy array
    sdf_hats = np.array(jnp.concatenate(sdf_hats, axis=1))

    return sdf_hats
