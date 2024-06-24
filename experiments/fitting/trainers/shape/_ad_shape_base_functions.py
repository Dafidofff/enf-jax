import wandb
from functools import partial

# from xvfbwrapper import Xvfb
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

from experiments.fitting.trainers._base._metrics import iou
from experiments.fitting.trainers.shape.utils.extract_mesh import extract_mesh_from_neural_field


class AutodecodingShapeBaseFunctions:

    def __init__(self):
        pass

    def iou_from_signed_distance(self, sdf_hat, sdf):
        """ Compute the intersection over union metric.

        Args:
            points: The points to evaluate.
            occ: The ground truth occupancy.

        Returns:
            float: The intersection over union metric.
        """
        return iou(sdf_hat < 0, sdf < 0).mean()

    def visualize_and_log(self, state, batch, p, a, window, name='recon'):
        """ Visualize and log the results.

        Args:
            state: The current training state.
        """
        # Unpack batch
        coords, sdf, _, shape_idx = batch

        # Forward through model and compute iou
        sdf_hat = self.nef.apply(state.params['nef'], coords, p=p[0:1], a=a[0:1],
                                 gaussian_window_size=window[0:1])
        iou_batch = self.iou_from_signed_distance(sdf_hat, sdf)

        # plot the points in 3d coloured by their sdf
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], c=sdf_hat[0], cmap='coolwarm')
        cbar = plt.colorbar(ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], c=sdf_hat[0], cmap='coolwarm'))
        cbar.set_label('SDF')
        wandb.log({f'{name}/pred_sdf': wandb.Image(fig)})

        # plot the points in 3d coloured by their sdf
        fig = plt.figure()
        below_0_idx = np.where(sdf_hat[0] <= 0)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[0, below_0_idx, 0], coords[0, below_0_idx, 1], coords[0, below_0_idx, 2], c=sdf[0, below_0_idx], cmap='coolwarm')
        cbar = plt.colorbar(ax.scatter(coords[0, below_0_idx, 0], coords[0, below_0_idx, 1], coords[0, below_0_idx, 2], c=sdf_hat[0, below_0_idx], cmap='coolwarm'))
        cbar.set_label('SDF')
        wandb.log({f'{name}/pred_sfd_surf': wandb.Image(fig)})

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], c=sdf[0], cmap='coolwarm')
        cbar = plt.colorbar(ax.scatter(coords[0, :, 0], coords[0, :, 1], coords[0, :, 2], c=sdf[0], cmap='coolwarm'))
        cbar.set_label('SDF')
        wandb.log({f'{name}/gt_sdf': wandb.Image(fig)})

        # Get all points for which sdf is below 0
        below_0_idx = np.where(sdf[0] <= 0)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[0, below_0_idx, 0], coords[0, below_0_idx, 1], coords[0, below_0_idx, 2], c=sdf[0, below_0_idx], cmap='coolwarm')
        cbar = plt.colorbar(ax.scatter(coords[0, below_0_idx, 0], coords[0, below_0_idx, 1], coords[0, below_0_idx, 2], c=sdf[0, below_0_idx], cmap='coolwarm'))
        cbar.set_label('SDF')
        wandb.log({f'{name}/gt_sdf_surf': wandb.Image(fig)})

        plt.close('all')


        # Extract meshes from predictions and visualise with blender
        mesh = extract_mesh_from_neural_field(
            apply_model_fn=partial(self.nef.apply, state.params['nef'], p=p[0:1], a=a[0:1],
                                   gaussian_window_size=window[0:1]),
            points_batch_size=self.config.training.max_num_sampled_points
        )

        # Add point location for latent points, with color, repeat over batch dimension
        colors = np.array([[1., 0, 0]])
        # ps = trimesh.points.PointCloud(p[i], colors=np.repeat(colors, p.shape[1], axis=0))
        mesh = mesh.scene()
        # mesh.add_geometry(ps)

        fname = self.config.logging.log_dir + f'/tmp-shape.obj'
        mesh.export(fname)
        wandb.log({
            f'{name}/shape': wandb.Object3D(open(fname),
                                                caption=f'{name}/shape_{shape_idx}_{iou_batch:.4f}_iou')})
        