import matplotlib.pyplot as plt
import warnings
import numpy as np
import wandb

import jax.numpy as jnp

from experiments.fitting.trainers._base._metrics import psnr


class AutodecodingImageBaseFunctions:
    """Base class for autodecoding image-based trainers. This implements a number of methods and functions that are
    shared between different autodecoding image-based trainers.

    Contains functions for visualization and logging of images.
    """
    def __init__(self, coords, **kwargs):
        self.coords = coords

    def calculate_psnr(self, gt, pred):
        """Calculate the PSNR between the ground truth and the prediction.

        Args:
            gt: The ground truth image. [batch, height, width, channels]
            pred: The predicted image. [batch, height, width, channels]
        """
        psnr_val = psnr(pred, gt, jnp.mean(pred, axis=(0, 1, 2)), jnp.std(pred, axis=(0, 1, 2)))
        return psnr_val

    def visualize_and_log(self, gt, state, p, a, window, name='recon'):
        """ Visualize and log the results.

        Args:
            state: The current training state.
        """

        # Broadcast coords over batch dimension
        coords = jnp.broadcast_to(self.coords, (p.shape[0], *self.coords.shape))

        out_all = None
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points):
            out = self.apply_enf_jitted(
                state.params['enf'], coords[:, i:i + self.config.training.max_num_sampled_points], p, a, window
            )

            if i == 0:
                out_all = out
            else:
                out_all = jnp.concatenate((out_all, out), axis=1)

        # Translate the poses
        p_trans = np.copy(p)
        p_trans[:, :, 0:2] += 0.3
        out_all_trans = None
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points):
            out = self.apply_enf_jitted(
                state.params['enf'], coords[:, i:i + self.config.training.max_num_sampled_points], p_trans, a, window
            )

            if i == 0:
                out_all_trans = out
            else:
                out_all_trans = jnp.concatenate((out_all_trans, out), axis=1)

        # Rotate the poses
        theta = jnp.pi / 3.5
        p_rot = np.copy(p)
        p_rot[:, :, 0:2] = p_rot[:, :, 0:2] @ jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

        if p_rot.shape[-1] == 3:
            p_theta = p_rot[:, :, 2]  # Obtain theta
            p_rot[:, :, 2] = p_theta - theta  # Update the orientation

        out_all_rot = None
        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points):
            out = self.apply_enf_jitted(
                state.params['enf'], coords[:, i:i + self.config.training.max_num_sampled_points], p_rot, a, window
            )

            if i == 0:
                out_all_rot = out
            else:
                out_all_rot = jnp.concatenate((out_all_rot, out), axis=1)

        # Randomly select a latent
        selected_idx = np.random.randint(low=0, high=p.shape[1])
        out_all_single = None

        # Loop over all coords in chunks of self.config.training.max_num_sampled_pixels
        for i in range(0, coords.shape[1], self.config.training.max_num_sampled_points):
            out = self.apply_enf_jitted(
                state.params['enf'],
                coords[:, i:i + self.config.training.max_num_sampled_points], p[:, selected_idx:selected_idx + 1],
                a[:, selected_idx:selected_idx + 1], window[:, selected_idx:selected_idx + 1]
            )

            if i == 0:
                out_all_single = out
            else:
                out_all_single = jnp.concatenate((out_all_single, out), axis=1)

        # Reshape to image
        out_all = out_all.reshape(-1, *self.config.dataset.image_shape)
        out_all_rot = out_all_rot.reshape(-1, *self.config.dataset.image_shape)
        out_all_trans = out_all_trans.reshape(-1, *self.config.dataset.image_shape)
        out_all_single = out_all_single.reshape(-1, *self.config.dataset.image_shape)
        gt = gt.reshape(-1, *self.config.dataset.image_shape)

        # Calculate PSNR, and log the images
        psnr = self.calculate_psnr(gt, out_all)
        self.cur_val_metric = psnr

        # Plot the first n images
        for i, img in enumerate(out_all):
            if i >= self.num_logged_samples:
                break

            fig, ax = plt.subplots(1, 6, figsize=(18, 3))

            with warnings.catch_warnings():
                ax[0].imshow(np.clip(gt[i], 0, 1))  # gt
                ax[1].imshow(np.clip(out_all[i], 0, 1))  # recon
                ax[2].imshow(np.clip(out_all[i], 0, 1))  # poses
                ax[2].scatter((p[i, :, 1] + 1) * self.config.dataset.image_shape[0] / 2,
                              (p[i, :, 0] + 1) * self.config.dataset.image_shape[1] / 2, c='r')
                ax[2].scatter((p[i, selected_idx, 1] + 1) * self.config.dataset.image_shape[0] / 2,
                              (p[i, selected_idx, 0] + 1) * self.config.dataset.image_shape[1] / 2, c='g')
                ax[3].imshow(np.clip(out_all_rot[i], 0, 1))  # rot
                ax[4].imshow(np.clip(out_all_trans[i], 0, 1))  # transl
                ax[5].imshow(np.clip(out_all_single[i], 0, 1))  # single random latent

            # Disable axis
            for a in ax:
                a.axis('off')

            # Set titles
            ax[0].set_title('GT')
            ax[1].set_title(f'Recon: {psnr[i]:.2f} dB')
            ax[2].set_title('Poses')
            ax[3].set_title('Rotated')
            ax[4].set_title('Translated')
            ax[5].set_title('Single')

            # If we have orientations in the poses, plot those.
            if p.shape[2] > 2:
                for j in range(p.shape[1]):
                    y = (p[i, j, 0] + 1) * self.config.dataset.image_shape[0] / 2
                    x = (p[i, j, 1] + 1) * self.config.dataset.image_shape[1] / 2
                    dy = jnp.cos(p[i, j, 2])
                    dx = jnp.sin(p[i, j, 2])
                    ax[2].quiver(x, y, dx, dy, angles='xy', scale_units='xy', color='r')

            wandb.log({f'{name}_{i}': wandb.Image(fig)}, commit=True)

        plt.close('all')
