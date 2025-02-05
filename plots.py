import matplotlib.pyplot as plt
import os
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting.base import GenericPlot
import numpy as np
import seaborn as sns
import torch
import math
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from captum.attr import visualization as viz
from ccbdl.config_loader.loaders import ConfigurationLoader
from matplotlib.colors import LinearSegmentedColormap
# from torcheval.metrics import FrechetInceptionDistance
from fid_custom import FrechetInceptionDistance
from utils import load_classifier_rgb, load_classifier, load_gan, ImageTensorDataset, load_cgan
from utils import attribution_maps_discriminator, attribution_maps_classifier, grayscale_to_rgb
from sklearn.metrics import confusion_matrix
from ignite.metrics import PSNR, SSIM
from ccbdl.evaluation.plotting import images
from ccbdl.network.nlrl import NLRL_AO
from setsize import set_size

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
        

class Loss_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the loss plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the loss plots.

        Returns
        -------
        fig of the loss plot, name of the loss plot (for saving the plot with this name).

        """
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_loss")
        yatr = self.learner.data_storage.get_item("a_train_loss")
        yt = self.learner.data_storage.get_item("test_loss")

        figs = []
        names = []
        
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathcal{L}_{\\mathrm{train}}$')
        ax.plot(x, yatr, label='$\\mathcal{L}_{\\mathrm{train\\_avg}}$')
        ax.plot(x, yt, label='$\\mathcal{L}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel('$\\mathcal{L}$', fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(ytr))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.set_yticks(range(1, 3, 1))
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "losses"))
        return figs, names


class TimePlot(GenericPlot):
    def __init__(self, learner):
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        if self.learner.learner_config["noise_prediction_model"] == 'conditional_unet':
            ax.plot(xs, [y - ys[0]for y in ys], label="conditional_unet_train_time")
        else:
            ax.plot(xs, [y - ys[0]for y in ys], label="unet_train_time")
        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$t$', fontsize=14)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(ys))
        ax.legend()
        
        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "time_plot"))
        return figs, names


class Fid_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the fid plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Fid_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot per epoch")
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the classifier
            classifier = load_classifier_rgb(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the classifier
            classifier = load_classifier_rgb(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
        
        self.fid_metric = FrechetInceptionDistance(model=classifier, feature_dim=10, device=self.learner.device)
        
        self.fid_scores = []

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def calculate_fid_score(self, real_images, generated_images, epochs):
        """
        Method to calculate the fid scores using real and generated images.

        Parameters
        ----------
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        generated_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        epochs : number of epochs run.

        Returns
        -------
        fid scores.

        """
        for epoch in epochs:
            # Convert real and fake images from grayscale to RGB
            real_images_rgb = grayscale_to_rgb(real_images[epoch])
            generated_images_rgb = grayscale_to_rgb(generated_images[epoch])

            # Update the metric for real images and generated images
            self.fid_metric.update(real_images_rgb, is_real=True)
            self.fid_metric.update(generated_images_rgb, is_real=False)
            
            fid_score = self.fid_metric.compute()
            self.fid_scores.append(fid_score)
        
        return self.fid_scores

    def plot(self):
        """
        method to plot the fid plot

        Returns
        -------
        fig of the fid plot, name of the fid plot (for saving the plot with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        real_images = self.learner.data_storage.get_item("real_images")
        generated_images = self.learner.data_storage.get_item("fake_images")
        
        fid_scores = self.calculate_fid_score(real_images, generated_images, epochs)
        
        fid_scores_np = [score.cpu().numpy() for score in fid_scores]  # Convert each score to numpy array

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(epochs, fid_scores_np, label="CNN_RGB")
        ax.set_xlabel("epochs")
        ax.set_ylabel("fid score")
        ax.set_xticks(range(len(epochs)))
        ax.set_title("fid score vs epoch")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("fid_scores")]

        return figs, names


class Psnr_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the psnr plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Psnr_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr plot")
        
        self.psnr_scores = []
        # Initialize the PSNR metric from ignite
        self.psnr_metric = PSNR(data_range=1.0, device=learner.device)

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def calculate_psnr_score(self, real_images, generated_images, epochs):
        """
        Method to calculate the psnr scores using real and generated images.

        Parameters
        ----------
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        generated_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].        
        epochs : number of epochs run.

        Returns
        -------
        psnr scores.

        """
        for epoch in epochs:
            # Resetting the PSNR metric for each epoch
            self.psnr_metric.reset()
            # Update the metric with real and generated images
            self.psnr_metric.update((generated_images[epoch], real_images[epoch]))
            # Compute the PSNR for the current epoch
            psnr = self.psnr_metric.compute()
            self.psnr_scores.append(psnr)
        
        return self.psnr_scores
    
    def plot(self):
        """
        method to plot the psnr plot

        Returns
        -------
        fig of the psnr plot, name of the psnr plot (for saving the plot with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        real_images = self.learner.data_storage.get_item("real_images")
        generated_images = self.learner.data_storage.get_item("fake_images")
        
        psnr_scores = self.calculate_psnr_score(real_images, generated_images, epochs)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epochs, psnr_scores, label="CNN_RGB")
        ax.set_xlabel("epochs")
        ax.set_ylabel("psnr score")
        ax.set_xticks(range(len(epochs)))
        ax.set_title("psnr score vs epoch")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("psnr_scores")]

        return figs, names


class Ssim_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the ssim plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Ssim_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating ssim plot")
        
        self.ssim_metric = SSIM(data_range=1.0, device=learner.device) # Initialize the SSIM metric from ignite
        self.ssim_scores = []
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def calculate_ssim_score(self, real_images, generated_images, epochs):
        """
        Method to calculate the ssim scores using real and generated images.

        Parameters
        ----------
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        generated_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        epochs : number of epochs run.

        Returns
        -------
        ssim scores.

        """
        for epoch in epochs:
            # Reset the SSIM metric for each epoch
            self.ssim_metric.reset()
            # Update the metric with the real and generated images
            self.ssim_metric.update((generated_images[epoch], real_images[epoch]))
            # Compute the SSIM for the current batch
            ssim = self.ssim_metric.compute()
            self.ssim_scores.append(ssim)
        
        return self.ssim_scores

    def plot(self):
        """
        method to plot the ssim plot

        Returns
        -------
        fig of the ssim plot, name of the ssim plot (for saving the plot with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        real_images = self.learner.data_storage.get_item("real_images")
        generated_images = self.learner.data_storage.get_item("fake_images")
        
        ssim_scores = self.calculate_ssim_score(real_images, generated_images, epochs)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epochs, ssim_scores, label="CNN_RGB")
        ax.set_xlabel("epochs")
        ax.set_ylabel("ssim score")
        ax.set_xticks(range(len(epochs)))
        ax.set_title("ssim score vs epoch")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("ssim")]

        return figs, names


class Noisy_image_generation(GenericPlot):
    def __init__(self, learner):
        """
        init method of the noisy image generation plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Noisy_image_generation, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def grid_2d(self, imgs, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):        
            ax = plt.subplot(rows, cols, i+1)
            # get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            
            ax.set_title(f"{i}")
            # remove axes
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
            
            # show img
            plt.imshow(img, cmap='gray')
        plt.tight_layout()
        
        return fig
    
    def noisy_images(self, ddpm, imgs, device):
        """
        method to create the noisy images on the dataset's images.

        Parameters
        ----------
        ddpm : ddpm or cddpm model.        
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        device : the device that was set for the model and other parameters in learner class.      
        labels (torch.Tensor): Tensor with N number of int values randomly from 0 to 9. Defaults to None.

        Returns
        -------
        noised_imgs : TYPE
            DESCRIPTION.

        """
        noised_imgs = []
        # Showing the forward process
        for percent in [0.1, 0.25, 0.5, 0.75, 1]:
            x = ddpm.noising_images(imgs.to(device),
                                    [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))])
            noised_imgs.append(x)
            
        return noised_imgs
    
    def plot(self):
        """
        method to plot the noisy images' plots

        Returns
        -------
        figs of the noisy images' plots, names of the noisy images' plots (for saving the plots with this name).

        """
        real_images = self.learner.data_storage.get_item("real_images")
        epochs = self.learner.data_storage.get_item("epochs_gen")
            
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(real_images)/total)

        figs = []
        names = []
        
        percent = [10, 25, 50, 75, 100]
        
        for types in ["train", "test"]:
            if types == "test":
                real_images = self.learner.data_storage.get_item("real_images_test")
            else:
                real_images = real_images

            for idx in range(total):
                # Calculate the index for the last batch of the current epoch
                last_batch_index = ((idx + 1) * batches_per_epoch) - 1
                
                real_images_per_epoch = real_images[last_batch_index]
                num = real_images_per_epoch.size(0)
                
                noised_images = self.noisy_images(self.learner.ddpm, 
                                                       real_images_per_epoch,
                                                       self.learner.device
                                                       )
                
                self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
                
                epoch = epochs[idx]
                
                for i in range(len(noised_images)):
                    figs.append(self.grid_2d(imgs=noised_images[i],
                                               **self.style))
                    
                    if types == "test":
                        names.append(os.path.join("noised_images", "test", f"epoch_{epoch}", f"percentage_{percent[i]}%"))
                    else:
                        names.append(os.path.join("noised_images", "train", f"epoch_{epoch}", f"percentage_{percent[i]}%"))
            
        return figs, names


class Image_generation(GenericPlot):
    def __init__(self, learner):
        """
        init method of the image generation plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Image_generation, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def grid_2d(self, imgs, noises, labels=None, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):
            ax1 = plt.subplot(rows, 2 * cols, 2 * i + 1)
            ax2 = plt.subplot(rows, 2 * cols, 2 * i + 2)

            # Get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise = noises[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise_values = noises[i].cpu().detach().numpy()
            
            if labels is not None:
                label = labels[i]

            # Plot noise
            ax1.imshow(noise, cmap='gray')
            ax1.set_title(f"mn: {noise_values.mean():.2f}\nsm: {noise_values.sum():.2f}")
            
            ax2.imshow(img, cmap='gray')
            if labels is not None:
                ax2.set_title(f"{i}\nlabel: {label}")
            else:
                ax2.set_title(f"{i}")

            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
            
            fig.suptitle("mn: mean of the noise tenosr used\nsn: sum of the noise tenosr used")

        plt.tight_layout()
        
        return fig
    
    def grid_2d_ldm(self, noises_latent, imgs_de, labels=None, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs_de.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs_de)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):
            ax = plt.subplot(rows, 2 * cols, 2 * i + 1)

            # Get contents
            img_de = imgs_de[i].cpu().detach().permute(1, 2, 0).squeeze()
            
            noise_values_latent = noises_latent[i].cpu().detach().numpy()
            
            if labels is not None:
                label = labels[i]

            # Plots
            ax.imshow(img_de, cmap='gray')
            if labels is not None:
                ax.set_title(f"{i},\nlabel: {label},\nmn: {noise_values_latent.mean():.2f}\nsm: {noise_values_latent.sum():.2f}")
            else:
                ax.set_title(f"{i},\nmn: {noise_values_latent.mean():.2f}\nsm: {noise_values_latent.sum():.2f}")

            # Remove axes
            ax.axis('off')
            
            fig.suptitle("mn: mean of the noise tenosr used\nsn: sum of the noise tenosr used")

        plt.tight_layout()
        
        return fig

    def plot(self):
        """
        method to plot the generated images' plots.

        Returns
        -------
        figs of the generated images' plots, names of the generated images' plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")      
        total = len(epochs)

        figs = []
        names = []
        
        for types in ["fixed_noise", "random_noise"]:
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':
                if types == "fixed_noise":
                    generated_images = self.learner.data_storage.get_item("generated_images_fixed")
                    noises = self.learner.data_storage.get_item("fixed_noise")
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.data_storage.get_item("gen_labels")
                else:
                    generated_images = self.learner.data_storage.get_item("generated_images_random")
                    noises = self.learner.data_storage.get_item("random_noise")
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.data_storage.get_item("gen_labels")
            else:
                if types == "fixed_noise":
                    noises_latent = self.learner.data_storage.get_item("fixed_noise_latent")
                    generated_images_de = self.learner.data_storage.get_item("generated_images_fixed")
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.data_storage.get_item("gen_labels")
                else:
                    noises_latent = self.learner.data_storage.get_item("random_noise_latent")
                    generated_images_de = self.learner.data_storage.get_item("generated_images")
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.data_storage.get_item("gen_labels")
             
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':    
                for idx in range(total):
                    generated_images_per_epoch = generated_images[idx]
                    noises_per_epoch = noises[idx]
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels_per_epoch = labels[idx]
                    num = generated_images_per_epoch.size(0)
                    
                    self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
                    
                    epoch = epochs[idx]
                    
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        figs.append(self.grid_2d(imgs=generated_images_per_epoch,
                                                 noises=noises_per_epoch,
                                                 labels=labels_per_epoch,
                                                   **self.style))
                    else:
                        figs.append(self.grid_2d(imgs=generated_images_per_epoch,
                                                 noises=noises_per_epoch,
                                                   **self.style))
                    
                    if types == "fixed_noise":
                        names.append(os.path.join("generated_images", "fixed_noise", f"epoch_{epoch}"))
                    else:
                        names.append(os.path.join("generated_images", "random_noise", f"epoch_{epoch}"))
            else:
                for idx in range(total):
                    noises_latent_per_epoch = noises_latent[idx]
                    generated_images_de_per_epoch = generated_images_de[idx]
                    
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels_per_epoch = labels[idx]
                    num = generated_images_de_per_epoch.size(0)
                    
                    self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
                    
                    epoch = epochs[idx]
                    
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        figs.append(self.grid_2d_ldm(noises_latent=noises_latent_per_epoch,
                                                     imgs_de=generated_images_de_per_epoch,                                                   
                                                     labels=labels_per_epoch,
                                                     **self.style))
                    else:
                        figs.append(self.grid_2d_ldm(noises_latent=noises_latent_per_epoch,
                                                     imgs_de=generated_images_de_per_epoch,
                                                     **self.style))
                    
                    if types == "fixed_noise":
                        names.append(os.path.join("generated_images", "fixed_noise", f"epoch_{epoch}"))
                    else:
                        names.append(os.path.join("generated_images", "random_noise", f"epoch_{epoch}"))
            
        return figs, names


class Image_generation_train_test(GenericPlot):
    def __init__(self, learner):
        """
        init method of the image generation plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Image_generation_train_test, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def grid_2d(self, origs, imgs, noises, labels=None, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        origs (torch.Tensor): Tensor with N original imgs with the shape [N x C x H x W].
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num_images = len(imgs)
        total_subplots = num_images * 3
        cols = int(math.ceil(math.sqrt(total_subplots / 3))) * 3  # Ensure cols are a multiple of 3
        rows = int(math.ceil(total_subplots / cols))
        
        fig = plt.figure(figsize = figsize)
        subplot_idx = 1
        for i in range(num_images):
            ax1 = plt.subplot(rows, cols, subplot_idx)
            ax2 = plt.subplot(rows, cols, subplot_idx + 1)
            ax3 = plt.subplot(rows, cols, subplot_idx + 2)

            # Get contents
            orig = origs[i].cpu().detach().permute(1, 2, 0).squeeze()
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise = noises[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise_values = noises[i].cpu().detach().numpy()
            
            if labels is not None:
                label = labels[i]

            ax1.imshow(orig, cmap='gray')
            if labels is not None:
                ax1.set_title(f"{i}\nlabel: {label}")
            else:
                ax1.set_title(f"{i}")
            
            ax2.imshow(noise, cmap='gray')
            ax2.set_title(f"mn: {noise_values.mean():.2f}\nsn: {noise_values.sum():.2f}")
            
            ax3.imshow(img, cmap='gray')
            ax3.set_title(f"denoised\n{i}")

            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            
            fig.suptitle("mn: mean of the noise tenosr used\nsn: sum of the noise tenosr used")
            
            subplot_idx += 3

        plt.tight_layout()
        
        return fig
    
    def grid_2d_ldm(self, noises_latent, origs, imgs, labels=None, figsize=(10, 10)):
        """
        Function to create reconstructions of image data.
    
        Parameters
        ----------
        origs_latent (torch.Tensor): Tensor with N original latent images with shape [N x C x H x W].
        imgs_latent (torch.Tensor): Tensor with N denoised latent images with shape [N x C x H x W].
        noises_latent (torch.Tensor): Tensor with N noise tensors corresponding to latent images with shape [N x C x H x W].
        origs (torch.Tensor): Tensor with N original images with shape [N x C x H x W].
        imgs (torch.Tensor): Tensor with N denoised images with shape [N x C x H x W].
        labels (list, optional): List of labels for images. Defaults to None.
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).
    
        Returns
        -------
        fig: matplotlib Figure object
            Figure containing the grid of images.
        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):
            ax1 = plt.subplot(rows, 2 * cols, 2 * i + 1)
            ax2 = plt.subplot(rows, 2 * cols, 2 * i + 2)
    
            # Get contents
            orig = origs[i].cpu().detach().permute(1, 2, 0).squeeze()
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
    
            noise_latent_values = noises_latent[i].cpu().detach().numpy()
    
            if labels is not None:
                label = labels[i]
    
            ax1.imshow(orig, cmap='gray')
            if labels is not None:
                ax1.set_title(f"{i}, label: {label}")
            else:
                ax1.set_title(f"{i}, original")
            
            ax2.imshow(img, cmap='gray')
            ax2.set_title(f"{i}, denoised,\nmn: {noise_latent_values.mean():.2f}\nsn: {noise_latent_values.sum():.2f}")
    
            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
    
            fig.suptitle("mn: mean of the noise tensor used\nsn: sum of the noise tensor used")
    
        plt.tight_layout()
    
        return fig

    def plot(self):
        """
        method to plot the generated images' plots.

        Returns
        -------
        figs of the generated images' plots, names of the generated images' plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")      
        total = len(epochs)

        figs = []
        names = []
        
        for types in ["train", "test"]:
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':
                if types == "train":
                    original_images = self.learner.data_storage.get_item("real_images")
                    fake_images = self.learner.data_storage.get_item("fake_images")
                    noises = self.learner.data_storage.get_item("noise_images")
                    generated_images = self.learner.data_storage.get_item("generated_images")
                    noise_tensors = self.learner.data_storage.get_item("noise_tensors")
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.data_storage.get_item("labels")
                else:
                    original_images = self.learner.data_storage.get_item("real_images_test")
                    fake_images = self.learner.data_storage.get_item("fake_images_test")
                    noises = self.learner.data_storage.get_item("noise_images_test")
                    generated_images = self.learner.data_storage.get_item("generated_images_test")
                    noise_tensors = self.learner.data_storage.get_item("noise_tensors_test")
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.data_storage.get_item("labels_test")
            else:
                if types == "train":
                    original_images = self.learner.data_storage.get_item("real_images")
                    fake_images = self.learner.data_storage.get_item("fake_images")
                    noises_latent = self.learner.data_storage.get_item("noisy_images_latent")
                    generated_images = self.learner.data_storage.get_item("generated_images")
                    noise_tensors_latent = self.learner.data_storage.get_item("noise_tensors_latent")
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.data_storage.get_item("labels")
                else:
                    original_images = self.learner.data_storage.get_item("real_images_test")
                    fake_images = self.learner.data_storage.get_item("fake_images_test")
                    noises_latent = self.learner.data_storage.get_item("noisy_images_test_latent")
                    generated_images = self.learner.data_storage.get_item("generated_images_test")
                    noise_tensors_latent = self.learner.data_storage.get_item("noise_tensors_test_latent")
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.data_storage.get_item("labels")
            
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':
                for typ in ["fake", "generated"]:
                    for idx in range(total):
                        if typ == "fake":
                            fake_images_per_epoch = fake_images[idx]
                            noises_per_epoch = noises[idx]
                        else:
                            generated_images_per_epoch = generated_images[idx]
                            noise_tensors_per_epoch = noise_tensors[idx]
                        original_images_per_epoch = original_images[idx]
                        
                        if self.learner.learner_config["diffusion_model"] == 'cddpm':
                            labels_per_epoch = labels[idx]
                            
                        num = fake_images_per_epoch.size(0)
                        
                        self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (25, 25)})
                        
                        epoch = epochs[idx]
                        
                        if typ == "fake":
                            if self.learner.learner_config["diffusion_model"] == 'cddpm':
                                figs.append(self.grid_2d(origs=original_images_per_epoch,
                                                         imgs=fake_images_per_epoch,
                                                         noises=noises_per_epoch,
                                                         labels=labels_per_epoch,
                                                           **self.style))
                            else:             
                                figs.append(self.grid_2d(origs=original_images_per_epoch,
                                                         imgs=fake_images_per_epoch,
                                                         noises=noises_per_epoch,
                                                           **self.style))
                            
                            if types == "train":
                                names.append(os.path.join("denoised_images", "train", f"epoch_{epoch}"))
                            else:
                                names.append(os.path.join("denoised_images", "test", f"epoch_{epoch}"))
                        else:
                            if self.learner.learner_config["diffusion_model"] == 'cddpm':
                                figs.append(self.grid_2d(origs=original_images_per_epoch,
                                                         imgs=generated_images_per_epoch,
                                                         noises=noise_tensors_per_epoch,
                                                         labels=labels_per_epoch,
                                                           **self.style))
                            else:             
                                figs.append(self.grid_2d(origs=original_images_per_epoch,
                                                         imgs=generated_images_per_epoch,
                                                         noises=noise_tensors_per_epoch,
                                                           **self.style))
                            
                            if types == "train":
                                names.append(os.path.join("generated_images", "train", f"epoch_{epoch}"))
                            else:
                                names.append(os.path.join("generated_images", "test", f"epoch_{epoch}"))
            else:
                for typ in ["fake", "generated"]:
                    for idx in range(total):
                        if typ == "fake":
                            fake_images_per_epoch = fake_images[idx]
                            noises_latent_per_epoch = noises_latent[idx]
                        else:
                            generated_images_per_epoch = generated_images[idx]
                            noise_tensors_latent_per_epoch = noise_tensors_latent[idx]
                        original_images_per_epoch = original_images[idx]
                        if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                            labels_per_epoch = labels[idx]
                            
                        num = fake_images_per_epoch.size(0)
                        
                        self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
                        
                        epoch = epochs[idx]
                        
                        if typ == "fake":
                            if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                                figs.append(self.grid_2d_ldm(noises_latent=noises_latent_per_epoch,
                                                         origs=original_images_per_epoch,
                                                         imgs=fake_images_per_epoch,
                                                         labels=labels_per_epoch,
                                                         **self.style))
                            else:             
                                figs.append(self.grid_2d_ldm(noises_latent=noises_latent_per_epoch,
                                                         origs=original_images_per_epoch,
                                                         imgs=fake_images_per_epoch,
                                                         **self.style))
                            
                            if types == "train":
                                names.append(os.path.join("denoised_images", "train", f"epoch_{epoch}"))
                            else:
                                names.append(os.path.join("denoised_images", "test", f"epoch_{epoch}"))
                        else:
                            if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                                figs.append(self.grid_2d_ldm(noises_latent=noise_tensors_latent_per_epoch,
                                                         origs=original_images_per_epoch,
                                                         imgs=generated_images_per_epoch,
                                                         labels=labels_per_epoch,
                                                         **self.style))
                            else:             
                                figs.append(self.grid_2d_ldm(noises_latent=noise_tensors_latent_per_epoch,
                                                         origs=original_images_per_epoch,
                                                         imgs=generated_images_per_epoch,
                                                         **self.style))
                            
                            if types == "train":
                                names.append(os.path.join("generated_images", "train", f"epoch_{epoch}"))
                            else:
                                names.append(os.path.join("generated_images", "test", f"epoch_{epoch}"))
            
        return figs, names


class Image_generation_train_test_initial(GenericPlot):
    def __init__(self, learner):
        """
        init method of the image generation plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Image_generation_train_test_initial, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("image generation by generator for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def grid_2d(self, origs, imgs, noises, labels=None, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        origs (torch.Tensor): Tensor with N original imgs with the shape [N x C x H x W].
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num_images = len(imgs)
        total_subplots = num_images * 3
        cols = int(math.ceil(math.sqrt(total_subplots / 3))) * 3  # Ensure cols are a multiple of 3
        rows = int(math.ceil(total_subplots / cols))
        
        fig = plt.figure(figsize = figsize)
        subplot_idx = 1
        for i in range(num_images):
            ax1 = plt.subplot(rows, cols, subplot_idx)
            ax2 = plt.subplot(rows, cols, subplot_idx + 1)
            ax3 = plt.subplot(rows, cols, subplot_idx + 2)

            # Get contents
            orig = origs[i].cpu().detach().permute(1, 2, 0).squeeze()
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise = noises[i].cpu().detach().permute(1, 2, 0).squeeze()
            noise_values = noises[i].cpu().detach().numpy()
            
            if labels is not None:
                label = labels[i]

            ax1.imshow(orig, cmap='gray')
            if labels is not None:
                ax1.set_title(f"{i}\nlabel: {label}")
            else:
                ax1.set_title(f"{i}")
            
            ax2.imshow(noise, cmap='gray')
            ax2.set_title(f"mn: {noise_values.mean():.2f}\nsn: {noise_values.sum():.2f}")
            
            ax3.imshow(img, cmap='gray')
            ax3.set_title(f"denoised\n{i}")

            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            
            fig.suptitle("mn: mean of the noise tenosr used\nsn: sum of the noise tenosr used")
            
            subplot_idx += 3

        plt.tight_layout()
        
        return fig
    
    def grid_2d_ldm(self, noises_latent, origs, imgs, labels=None, figsize=(10, 10)):
        """
        Function to create reconstructions of image data.
    
        Parameters
        ----------
        origs_latent (torch.Tensor): Tensor with N original latent images with shape [N x C x H x W].
        imgs_latent (torch.Tensor): Tensor with N denoised latent images with shape [N x C x H x W].
        noises_latent (torch.Tensor): Tensor with N noise tensors corresponding to latent images with shape [N x C x H x W].
        origs (torch.Tensor): Tensor with N original images with shape [N x C x H x W].
        imgs (torch.Tensor): Tensor with N denoised images with shape [N x C x H x W].
        labels (list, optional): List of labels for images. Defaults to None.
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).
    
        Returns
        -------
        fig: matplotlib Figure object
            Figure containing the grid of images.
        """
    
        num_images = len(imgs)
        total_subplots = num_images * 2
        cols = int(math.ceil(math.sqrt(total_subplots / 2))) * 2
        rows = int(math.ceil(total_subplots / cols))
    
        fig = plt.figure(figsize=figsize)
        subplot_idx = 1
        for i in range(num_images):
            ax1 = plt.subplot(rows, cols, subplot_idx)
            ax2 = plt.subplot(rows, cols, subplot_idx + 1)
    
            # Get contents
            orig = origs[i].cpu().detach().permute(1, 2, 0).squeeze()
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
    
            noise_latent_values = noises_latent[i].cpu().detach().numpy()
    
            if labels is not None:
                label = labels[i]
    
            ax1.imshow(orig, cmap='gray')
            if labels is not None:
                ax1.set_title(f"{i}, label: {label}")
            else:
                ax1.set_title(f"{i}, original")
            
            ax2.imshow(img, cmap='gray')
            ax2.set_title(f"{i}, denoised,\nmn: {noise_latent_values.mean():.2f}\nsn: {noise_latent_values.sum():.2f}")
    
            # Remove axes
            ax1.axis('off')
            ax2.axis('off')
    
            fig.suptitle("mn: mean of the noise tensor used\nsn: sum of the noise tensor used")
    
            subplot_idx += 2
    
        plt.tight_layout()
    
        return fig

    def plot(self):
        """
        method to plot the generated images' plots.

        Returns
        -------
        figs of the generated images' plots, names of the generated images' plots (for saving the plots with this name).

        """

        figs = []
        names = []
        
        for types in ["train", "test"]:
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':
                if types == "train":
                    original_images = self.learner.original_images_initial
                    fake_images = self.learner.fake_images_initial
                    noises = self.learner.noises_initial
                    generated_images = self.learner.generated_images_initial
                    noise_tensors = self.learner.noise_tensors_initial
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.labels_initial
                else:
                    original_images = self.learner.original_images_test_initial
                    fake_images = self.learner.fake_images_test_initial
                    noises = self.learner.noises_test_initial
                    generated_images = self.learner.generated_images_test_initial
                    noise_tensors = self.learner.noises_test_initial
                    if self.learner.learner_config["diffusion_model"] == 'cddpm':
                        labels = self.learner.labels_test_initial
            else:
                if types == "train":
                    original_images = self.learner.original_images_initial
                    fake_images = self.learner.fake_images_initial
                    noises_latent = self.learner.noises_latent_initial
                    generated_images = self.learner.generated_images_initial
                    noise_tensors_latent = self.learner.noise_tensor_latent_initial
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.labels_initial
                else:
                    original_images = self.learner.original_images_test_initial
                    fake_images = self.learner.fake_images_test_initial
                    noises_latent = self.learner.noises_latent_test_initial
                    generated_images = self.learner.generated_images_test_initial
                    noise_tensors_latent = self.learner.noise_tensor_latent_test_initial
                    if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                        labels = self.learner.labels_initial_test
            
            if self.learner.learner_config["diffusion_model"] == 'ddpm' or self.learner.learner_config["diffusion_model"] == 'cddpm':
                for typ in ["fake", "generated"]:
                    num = original_images.size(0)
                    
                    self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (25, 25)})
                    if typ == "fake":
                        if self.learner.learner_config["diffusion_model"] == 'cddpm':
                            figs.append(self.grid_2d(origs=original_images,
                                                     imgs=fake_images,
                                                     noises=noises,
                                                     labels=labels,
                                                       **self.style))
                        else:             
                            figs.append(self.grid_2d(origs=original_images,
                                                     imgs=fake_images,
                                                     noises=noises,
                                                       **self.style))
                        
                        if types == "train":
                            names.append(os.path.join("denoised_images", "train", "initial"))
                        else:
                            names.append(os.path.join("denoised_images", "test", "initial"))
                    else:
                        if self.learner.learner_config["diffusion_model"] == 'cddpm':
                            figs.append(self.grid_2d(origs=original_images,
                                                     imgs=generated_images,
                                                     noises=noise_tensors,
                                                     labels=labels,
                                                       **self.style))
                        else:             
                            figs.append(self.grid_2d(origs=original_images,
                                                     imgs=generated_images,
                                                     noises=noise_tensors,
                                                       **self.style))
                        
                        if types == "train":
                            names.append(os.path.join("generated_images_initial", "train", "Initial"))
                        else:
                            names.append(os.path.join("generated_images_initial", "test", "Initial"))
            else:
                for typ in ["fake", "generated"]:                    
                    num = original_images.size(0)
                    
                    self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (15, 15)})
                    
                    if typ == "fake":
                        if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                            figs.append(self.grid_2d_ldm(noises_latent=noises_latent,
                                                     origs=original_images,
                                                     imgs=fake_images,
                                                     labels=labels,
                                                     **self.style))
                        else:             
                            figs.append(self.grid_2d_ldm(noises_latent=noises_latent,
                                                     origs=original_images,
                                                     imgs=fake_images,
                                                     **self.style))
                        
                        if types == "train":
                            names.append(os.path.join("denoised_images", "train", "Initial"))
                        else:
                            names.append(os.path.join("denoised_images", "test", "Initial"))
                    else:
                        if self.learner.learner_config["diffusion_model"] == 'conditional_latent_dm':
                            figs.append(self.grid_2d_ldm(noises_latent=noise_tensors_latent,
                                                     origs=original_images,
                                                     imgs=generated_images,
                                                     labels=labels,
                                                     **self.style))
                        else:             
                            figs.append(self.grid_2d_ldm(noises_latent=noise_tensors_latent,
                                                     origs=original_images,
                                                     imgs=generated_images,
                                                     **self.style))
                        
                        if types == "train":
                            names.append(os.path.join("generated_images", "train", "Initial"))
                        else:
                            names.append(os.path.join("generated_images", "test", "Initial"))
            
        return figs, names


class Confusion_matrix_classifier(GenericPlot):
    def __init__(self, learner, **kwargs):
        """
        init method of the confusion matrix plots with respect to pre trained classifier.

        Parameters
        ----------
        learner : learner class.      
        **kwargs (TYPE): Additional Parameters for further tasks.

        Returns
        -------
        None.

        """
        super(Confusion_matrix_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating confusion matrix based on pre trained classifier")
        
        # get default stype values
        self.style = {}
        self.style["figsize"] = self.get_value_with_default("figsize", (12, 12), kwargs)
        self.style["cmap"] = self.get_value_with_default("cmap", "Blues", kwargs)
        self.style["ticks"] = self.get_value_with_default("ticks", "auto", kwargs)
        self.style["xrotation"] = self.get_value_with_default("xrotation", "vertical", kwargs)
        self.style["yrotation"] = self.get_value_with_default("yrotation", "horizontal", kwargs)
        self.style["color_threshold"] = self.get_value_with_default("color_threshold", 50, kwargs)
        self.style["color_high"] = self.get_value_with_default("color_high", "white", kwargs)
        self.style["color_low"] = self.get_value_with_default("color_low", "black", kwargs)
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def plot(self):
        """
        method to plot the confusion matrix plots.

        Returns
        -------
        figs of the confusion matrix plots, names of the confusion matrix plots (for saving the plots with this name).

        """
        process_imgs = Tsne_plot_classifier(self.learner)
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the classifier
            classifier = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the classifier
            classifier = load_classifier(layer='nlrl')
        else:
                raise ValueError("Invalid values, it's either linear or nlrl")
        
        # Setting concatenation false by initializing value as 0
        cat = 0
        
        names = []
        figs = []
        
        total_real_images = self.learner.data_storage.get_item("real_images")
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        total_true_labels = self.learner.data_storage.get_item("labels")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        total = len(epochs)
        
        for idx in range(total):
            epoch = epochs[idx]
            
            # Access the last batch of real, fake, and label data for the current epoch
            real_images = total_real_images[idx]
            fake_images = total_fake_images[idx]
            true_labels = total_true_labels[idx]
    
            # Process real images
            real_dataset = ImageTensorDataset(real_images)
            real_data_loader = DataLoader(real_dataset, batch_size=64, shuffle=False)
            _, real_predicted_labels = process_imgs.process_images(real_data_loader, classifier, cat)
            
            # Process fake images
            fake_dataset = ImageTensorDataset(fake_images)
            fake_data_loader = DataLoader(fake_dataset, batch_size=64, shuffle=False)
            _, fake_predicted_labels = process_imgs.process_images(fake_data_loader, classifier, cat)
            
            for types in ["real", "fake"]:
                predictions = real_predicted_labels if types == 'real' else fake_predicted_labels
                correct_labels = true_labels
                
                # Flatten the predictions list to a single tensor
                # Each tensor in the list is a single-element tensor, so we concatenate them and then flatten
                predictions_tensor = torch.cat(predictions, dim=0).flatten()
                
                # Move tensors to CPU and convert to numpy for sklearn compatibility
                predictions_np = predictions_tensor.cpu().numpy()
                correct_labels_np = correct_labels.cpu().numpy()
                
                figs.append(images.plot_confusion_matrix(predictions_np,
                                                         correct_labels_np,
                                                         **self.style))

                names.append(os.path.join("confusion_matrices", "classifier_training_based", f"{types}", f"epoch_{epoch}"))
        return figs, names


class Confusion_matrix_gan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the confusion matrix plots with respect to pre trained gan or conditional.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Confusion_matrix_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating confusion matrix for real vs. fake predictions")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the confusion matrix plots.

        Returns
        -------
        figs of the confusion matrix plots, names of the confusion matrix plots (for saving the plots with this name).

        """
        names = []
        figs = []
        
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        if self.learner.learner_config["diffusion_model"] == 'cddpm':
            labels = self.learner.data_storage.get_item("labels")
        
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        threshold = 0.5
        
        total = len(epochs)
        
        if self.learner.learner_config["diffusion_model"] == 'cddpm':
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                cgan = load_cgan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                cgan = load_cgan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")
                
            model = cgan.discriminator
        else:
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                gan = load_gan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                gan = load_gan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")

            model = gan.discriminator
        
        for idx in range(total):
            fig, ax = plt.subplots(figsize=(12, 8))
            epoch = epochs[idx]
            
            if self.learner.learner_config["diffusion_model"] == 'cddpm':
                real_probs = model(real_images[idx], labels[idx])
                fake_probs = model(fake_images[idx], labels[idx])
            else:
                real_probs = model(real_images[idx])
                fake_probs = model(fake_images[idx])
            
            real_probs = real_probs.detach().cpu().numpy()
            fake_probs = fake_probs.detach().cpu().numpy()
            
            # Extract and flatten the predictions for the current epoch
            fake_probs_per_epoch = fake_probs.flatten()
            real_probs_per_epoch = real_probs.flatten()

            # Convert probabilities to binary predictions using a threshold eg 0.5
            fake_predictions = (fake_probs_per_epoch < threshold).astype(int)
            real_predictions = (real_probs_per_epoch > threshold).astype(int)

            # Concatenate predictions and true labels
            predictions = np.concatenate([fake_predictions, real_predictions])
            correct_labels = np.concatenate([np.zeros_like(fake_predictions), np.ones_like(real_predictions)])
            
            # Compute confusion matrix
            matrix = confusion_matrix(correct_labels, predictions)

            # Plot the confusion matrix using seaborn's heatmap
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'])
            
            figs.append(fig)
            names.append(os.path.join("confusion_matrices", "gan_training_based", f"epoch_{epoch}"))
        return figs, names


class Tsne_plot_images_combined(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_images_combined, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating combined tsne plots")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the tsne plot.

        Returns
        -------
        fig of the tnse plot, name of the tsne plot (for saving the plot with this name).

        """
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        labels = self.learner.data_storage.get_item("labels")

        num_classes = 10
        # Create color map for real and fake images
        colors = cm.rainbow(np.linspace(0, 1, num_classes))

        tsne = TSNE(n_components=2)

        fig, ax = plt.subplots(figsize=(16, 10))

        # Storage for transformed data
        real_transformed_list = []
        fake_transformed_list = []

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx].view(real_images[idx].size(0), -1).cpu().numpy()  # real_images of current class
            fake_image = fake_images[idx].view(fake_images[idx].size(0), -1).cpu().numpy()  # generated_images of current class

            # Compute TSNE for each class and store
            real_transformed_list.append(tsne.fit_transform(real_image))
            fake_transformed_list.append(tsne.fit_transform(fake_image))

        # Plot each class for real and fake images
        for lb in range(num_classes):
            ax.scatter(real_transformed_list[lb][:, 0], real_transformed_list[lb][:, 1], c=[colors[lb]], label=f'{lb}', marker='o')
            ax.scatter(fake_transformed_list[lb][:, 0], fake_transformed_list[lb][:, 1], c=[colors[lb]], label=f'{lb}', marker='^')

        ax.set_title("Combined t-SNE plot for all labels")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()
        figs = [fig]
        names = [os.path.join("analysis_plots", "tsne_plots", "combined_tsne_plot")]
        return figs, names


class Tsne_plot_images_separate(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_images_separate, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots for each label")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the tsne plot.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        real_images = self.learner.data_storage.get_item("real_images")
        fake_images = self.learner.data_storage.get_item("fake_images")
        labels = self.learner.data_storage.get_item("labels")

        figs = []
        names = []

        num_classes = 10

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx]  # real_images of current class  
            fake_image = fake_images[idx] # generated_images of current class

            real_image = real_image.view(real_image.size(0), -1).cpu().detach().numpy()
            fake_image = fake_image.view(fake_image.size(0), -1).cpu().numpy()

            # compute TSNE
            tsne = TSNE(n_components=2)

            real_transformed = tsne.fit_transform(real_image)
            fake_transformed = tsne.fit_transform(fake_image)

            fig, ax = plt.subplots(figsize=(16, 10))

            ax.scatter(
                real_transformed[:, 0], real_transformed[:, 1], label='real images', marker='o')

            ax.scatter(
                fake_transformed[:, 0], fake_transformed[:, 1], label='fake images', marker='^')

            ax.set_title(f"t-SNE plot for label {lb}")
            ax.legend()

            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "seperate_plots", f"label_{lb}"))          
        return figs, names

    
class Tsne_plot_classifier(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained classifier.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features(self, classifier, imgs):
        """
        method to extract the features from a layer of the pre trained classifier.

        Parameters
        ----------
        classifier : pre trained classifier.       
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].

        Returns
        -------
        features from the classifier.

        """
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        if self.learner.learner_config["layer"] == 'nlrl':
            handle = classifier.model[-2].register_forward_hook(get_activation('conv'))
        else:
            handle = classifier.model[-1].register_forward_hook(get_activation('conv'))
            
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the classifier.

        Returns
        -------
        tsne results after computation.

        """
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat):
        """
        method to process the images from the data loader.

        Parameters
        ----------
        data_loader : the respective images' dataloader.       
        classifier : pre trained classifier.       
        cat (int): condition to check whether concatenation of features and labels required or not.

        Returns
        -------
        features and labels of the dataloader's images'.

        """
        all_features = []
        all_labels = []
        
        for imgs in data_loader:
            outputs = classifier(imgs)
            _, predicted_labels = torch.max(outputs, 1)
            features = self.get_features(classifier, imgs)
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the classifier
            classifier = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the classifier
            classifier = load_classifier(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
        
        # Setting concatenation true by initializing value as 1
        cat = 1
        config = ConfigurationLoader().read_config("config.yaml")
        data_config = config["data"]
    
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        real_images = total_real_images[-1]
        real_dataset = ImageTensorDataset(real_images)
        real_data_loader = DataLoader(real_dataset, batch_size=data_config["batch_size"], shuffle=False)
        real_features, real_labels = self.process_images(real_data_loader, classifier, cat)
    
        # Process generated images  
        total_generated_images = self.learner.data_storage.get_item("fake_images")
        generated_images = total_generated_images[-1]
        generated_dataset = ImageTensorDataset(generated_images)
        generated_data_loader = DataLoader(generated_dataset, batch_size=data_config["batch_size"], shuffle=False)
        generated_features, generated_labels = self.process_images(generated_data_loader, classifier, cat)
        
        real_label_counts = [torch.sum(real_labels == i).item() for i in range(10)]
        fake_label_counts = [torch.sum(generated_labels == i).item() for i in range(10)]
    
        # Combine features for t-SNE
        combined_features = torch.cat([real_features, generated_features], dim=0)
        tsne_results = self.compute_tsne(combined_features.cpu().numpy())
    
        # Split t-SNE results back into real and generated
        real_tsne = tsne_results[:len(real_features)]
        fake_tsne = tsne_results[len(real_features):]
        
        # Define a color palette for the labels
        palette = sns.color_palette("colorblind", 10)
        palette_fake = sns.color_palette("dark", 10)
    
        # Plotting
        figs, names = [], []
        for label in range(10):  # Mnist dataset has 10 labels
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (real_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label}",
                alpha=0.5,
                color=palette[label]
            )
            # Fake images scatter plot
            fake_indices = (generated_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label}", 
                marker="^",
                alpha=0.5,
                color=palette_fake[label]
            )
            ax.set_title(f"t-SNE visualization for label {label}, Counts - Real: {real_label_counts[label]}, Fake: {fake_label_counts[label]}")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_classifier", f"label_{label}"))   
        
        fig, ax = plt.subplots(figsize=(16, 10))       
        for label in range(10):  # Mnist dataset has 10 labels
            # Filter data points by label
            real_indices = (real_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label}", 
                color=palette[label],
                alpha=0.5
            )
            # Fake images scatter plot
            fake_indices = (generated_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label}",
                color=palette_fake[label],
                alpha=0.5
            )
        ax.set_title("t-SNE visualization for all labels")
        ax.legend()
        
        figs.append(fig)        
        names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_classifier", "combined"))
        plt.close(fig)
        return figs, names


class Tsne_plot_dis_gan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_dis_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on gan's discriminator's features and discriminator's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features_and_predictions(self, layer_num, images, discriminator):
        """
        method to extract the features from a layer of the pre trained discriminator.

        Parameters
        ----------
        layer_num (int): specifying the layer number of the discriminator from which the features are extracted.     
        images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset or generated ones.     
        discriminator : pre trained discriminator.

        Returns
        -------
        features from the discriminator.

        """
        features = []
        predictions = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator[layer_num].register_forward_hook(hook)

        # Process images through the discriminator
        for imgs in images:
            pred = discriminator(imgs)
            predictions.append(pred.detach())

        handle.remove()  # Remove the hook
        return torch.cat(features), torch.cat(predictions)
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the discriminator.

        Returns
        -------
        tsne results after computation.

        """
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)

        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images) / total)
        real_images = total_real_images[-batches_per_epoch:]

        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]

        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            gan = load_gan(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            gan = load_gan(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")

        discriminator = gan.discriminator
        
        if self.learner.learner_config["layer"] == 'nlrl':
            layer_num = -2
        else:
            layer_num = -3
        
        labels_cat = torch.cat(labels, dim=0)

        # Extract features and predictions from the discriminator for real and fake images
        real_features, real_predictions = self.get_features_and_predictions(layer_num, real_images, discriminator)
        fake_features, fake_predictions = self.get_features_and_predictions(layer_num, fake_images, discriminator)

        # Flatten the features
        real_features = real_features.view(real_features.size(0), -1)
        fake_features = fake_features.view(fake_features.size(0), -1)

        # Combine features for t-SNE
        combined_features = torch.cat([real_features, fake_features], dim=0)
        tsne_results = self.compute_tsne(combined_features)

        # Split t-SNE results back into real and fake
        half = len(tsne_results) // 2
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]

        # Prepare data for plotting
        real_pred_label = (real_predictions > 0.5).cpu().numpy().astype(int).flatten()
        fake_pred_label = (fake_predictions > 0.5).cpu().numpy().astype(int).flatten()

        # Combined plot for real and fake images
        fig, ax = plt.subplots(figsize=(28, 22))
        palette = sns.color_palette("colorblind", 10)
        
        # Plotting
        figs, names = [], []

        # Plot real images with different colors for each label and mark based on predictions
        for label in range(10):
            label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
            real_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

            correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]  # Indices of correctly classified real images
            incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]  # Indices of incorrectly classified real images

            sns.scatterplot(
                ax=ax,
                x=real_tsne[correct_real_indices, 0],
                y=real_tsne[correct_real_indices, 1],
                color=palette[label],
                marker='o',
                label=f"Real {label} (classified as real)",
                alpha=0.5
            )
            sns.scatterplot(
                ax=ax,
                x=real_tsne[incorrect_real_indices, 0],
                y=real_tsne[incorrect_real_indices, 1],
                color='blue',
                marker='o',
                label=f"Real {label} (classified as fake)",
                alpha=0.5
            )

        # Plot fake images
        correct_fake_indices = np.where(fake_pred_label == 0)[0]
        incorrect_fake_indices = np.where(fake_pred_label == 1)[0]

        sns.scatterplot(
            ax=ax,
            x=fake_tsne[correct_fake_indices, 0],
            y=fake_tsne[correct_fake_indices, 1],
            color='gray',
            marker='X',
            label="Fake (classified as fake)",
            alpha=0.5
        )
        sns.scatterplot(
            ax=ax,
            x=fake_tsne[incorrect_fake_indices, 0],
            y=fake_tsne[incorrect_fake_indices, 1],
            color='green',
            marker='X',
            label="Fake (classified as real)",
            alpha=0.5
        )

        ax.set_title("t-SNE visualization of Real and Fake Images")
        ax.legend(title='Predictions', loc='best')

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_gan_discriminator", "combined_plot"))

        # Separate plot for real images
        for label in range(10):
            fig, ax = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis
            label_indices = (labels_cat == label).cpu().numpy()  # Get a boolean array for images with the current label
            real_label_indices = np.where(label_indices)[0]  # Get indices of images with the current label

            correct_real_indices = real_label_indices[real_pred_label[real_label_indices] == 0]  # Indices of correctly classified real images
            incorrect_real_indices = real_label_indices[real_pred_label[real_label_indices] == 1]  # Indices of incorrectly classified real images

            sns.scatterplot(
                ax=ax,
                x=real_tsne[correct_real_indices, 0],
                y=real_tsne[correct_real_indices, 1],
                color=palette[label],
                marker='o',
                label=f"Real {label} (classified as real)",
                alpha=0.5
            )
            sns.scatterplot(
                ax=ax,
                x=real_tsne[incorrect_real_indices, 0],
                y=real_tsne[incorrect_real_indices, 1],
                color='blue',
                marker='o',
                label=f"Real {label} (classified as fake)",
                alpha=0.5
            )

            ax.set_title(f"t-SNE visualization of Real Images (Label {label}")
            ax.legend(title='Predictions', loc='best')

            figs.append(fig)
            plt.close(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_gan_discriminator", f"label_{label}_real"))

        # Separate plot for fake images
        fig, ax = plt.subplots(figsize=(16, 10))  # Re-initialize figure and axis

        sns.scatterplot(
            ax=ax,
            x=fake_tsne[correct_fake_indices, 0],
            y=fake_tsne[correct_fake_indices, 1],
            color='gray',
            marker='X',
            label="Fake (classified as fake)",
            alpha=0.5
        )
        sns.scatterplot(
            ax=ax,
            x=fake_tsne[incorrect_fake_indices, 0],
            y=fake_tsne[incorrect_fake_indices, 1],
            color='green',
            marker='X',
            label="Fake (classified as real)",
            alpha=0.5
        )

        ax.set_title("t-SNE visualization of Fake Images")
        ax.legend(title='Predictions', loc='best')

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_gan_discriminator", "fake"))
        return figs, names


class Tsne_plot_dis_cgan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained conditional discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_dis_cgan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on cgan's discriminator's features and discriminator's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features(self, layer_num, real_images, fake_images, discriminator, labels):
        """
        method to extract the features from a layer of the pre trained conditional discriminator.

        Parameters
        ----------
        layer_num (int): specifying the layer number of the discriminator from which the features are extracted.    
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.    
        fake_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].   
        discriminator : pre trained conditional discriminator.

        Returns
        -------
        features from the conditional discriminator.

        """
        features = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator.dis[layer_num].register_forward_hook(hook)

        # Process real images through the discriminator
        for imgs, lbls in zip(real_images, labels):
            discriminator(imgs, lbls)

        # Process fake images through the discriminator
        for imgs, lbls in zip(fake_images, labels):
            discriminator(imgs, lbls)

        handle.remove()  # Remove the hook
        return torch.cat(features)
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the conditional discriminator.

        Returns
        -------
        tsne results after computation.

        """
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images)/total)
        real_images = total_real_images[-batches_per_epoch:]
        
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]
        
        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            cgan = load_cgan(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            cgan = load_cgan(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
            
        discriminator = cgan.discriminator 
        
        if self.learner.learner_config["layer"] == 'nlrl':
            layer_num = -4
        else:
            layer_num = -4
           
        # Extract features from the discriminator
        features = self.get_features(layer_num, real_images, fake_images, discriminator, labels)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        labels = torch.cat(labels, dim=0)
        label_counts = [torch.sum(labels == i).item() for i in range(10)]
        
        # Compute t-SNE
        tsne_results = self.compute_tsne(features)
        
        half = len(tsne_results) // 2

        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]
        
        # Plotting
        figs, names = [], []
        for label in range(10):  # Mnist dataset has 10 labels
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label}", 
                alpha=0.5,
                marker="o"
            )
            # Fake images scatter plot
            fake_indices = (labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label}", 
                alpha=0.5,
                marker="^",
            )
            ax.set_title(f"t-SNE visualization for label {label}, count: {label_counts[label]}")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"label_{label}"))   
        return figs, names


class Attribution_plots_classifier(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained classifier

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for grayscale images")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, original_image, title, fig, ax, label, img_name, types, cmap):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if original_image is not None and len(original_image.shape) == 2:
                original_image = np.expand_dims(original_image, axis=2)
            # Call the visualization function with the squeezed attribute
            viz.visualize_image_attr(attr,
                                     original_image=np.squeeze(original_image),  # Squeeze the image as well
                                     method='heat_map',
                                     sign='all',
                                     show_colorbar=True,
                                     title=title,
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        names = []
        figs = []
        max_images_per_plot = 4  # Define a constant for the maximum number of images per plot
        
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            model = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            model = load_classifier(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
                
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        for types in ["real", "fake"]:
            inputs = real_images[-1].clone().detach().requires_grad_(True) if types == 'real' else fake_images[-1].clone().detach().requires_grad_(True) # Requires gradients set true
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Get attribution maps for different techniques
            saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_classifier(model, 
                                                                                                                                  inputs, 
                                                                                                                                  preds)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
            
            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for subset in subsets:
                num_rows = len(subset)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in subset:
                    img = np.squeeze(inputs[idx].cpu().detach().numpy())  # Squeeze the image to 2D
                    pred = preds[idx]
            
                    # Retrieve the attribution maps for the current image
                    results = [
                        np.squeeze(saliency_maps[idx].cpu().detach().numpy()),
                        np.squeeze(guided_backprop_maps[idx].cpu().detach().numpy()),
                        np.squeeze(input_x_gradient_maps[idx].cpu().detach().numpy()),
                        np.squeeze(deconv_maps[idx].cpu().detach().numpy()),
                        np.squeeze(occlusion_maps[idx].cpu().detach().numpy())
                    ]
            
                    # Display the original image
                    axs[count, 0].imshow(img, cmap='gray')
                    axs[count, 0].set_title(f"Predicted label: {pred}")
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        
                        # Call the visualization function, passing None for label and img_name since they are not applicable
                        self.safe_visualize(res, img, title, fig, axs[count, col + 1], None, None, types, cmap)
            
                    count += 1
            
                # Set the overall title for the figure based on the type of data
                fig.suptitle(f"Classifier's view on {types.capitalize()} Data and the respective Attribution maps")
            
                # Store the figure with an appropriate name
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("analysis_plots", "attribution_plots_classifier", f"{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names
    

class Attribution_plots_discriminator(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained discriminator or conditional discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_discriminator, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for some grayscale images of mnist")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, original_image, title, fig, ax, label, img_name, types, cmap):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if original_image is not None and len(original_image.shape) == 2:
                original_image = np.expand_dims(original_image, axis=2)
            # Call the visualization function with the squeezed attribute
            viz.visualize_image_attr(attr,
                                     original_image=np.squeeze(original_image),  # Squeeze the image as well
                                     method='heat_map',
                                     sign='all',
                                     show_colorbar=True,
                                     title=title,
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        # Plotting
        figs, names = [], []
        # Load the discriminator
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            gan = load_gan(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            gan = load_gan(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")

        model = gan.discriminator
        
        # Custom cmap for better visulaization
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")

        for types in ["real", "fake"]:
            inputs = real_images[-1].clone() if types == 'real' else fake_images[-1].clone()
    
            inputs.requires_grad = True  # Requires gradients set true
            
            preds = model(inputs)
            # Get attribution maps for different techniques
            saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_discriminator(model, inputs)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]

            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            subsets = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for subset in subsets:
                num_rows = len(subset)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in subset:
                    img = np.squeeze(inputs[idx].cpu().detach().numpy())  # Squeeze the image to 2D
                    pred = preds[idx].item()           
                    # Retrieve the attribution maps for the current image
                    results = [
                        np.squeeze(saliency_maps[idx].cpu().detach().numpy()),
                        np.squeeze(guided_backprop_maps[idx].cpu().detach().numpy()),
                        np.squeeze(input_x_gradient_maps[idx].cpu().detach().numpy()),
                        np.squeeze(deconv_maps[idx].cpu().detach().numpy()),
                        np.squeeze(occlusion_maps[idx].cpu().detach().numpy())
                    ]
            
                    # Display the original image
                    axs[count, 0].imshow(img, cmap='gray')
                    axs[count, 0].set_title(f"Prediction: {pred:.3f}")
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        
                        # Call the visualization function, passing None for label and img_name since they are not applicable
                        self.safe_visualize(res, img, title, fig, axs[count, col + 1], None, None, types, cmap)
            
                    count += 1
            
                # Set the overall title for the figure based on the type of data
                fig.suptitle(f"Discriminator's view on {types.capitalize()} Data and the respective Attribution maps")
            
                # Store the figure with an appropriate name
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("analysis_plots", "attribution_plots", f"{types}_subsets_{subsets.index(subset) + 1}"))
        return figs, names


class Hist_plot_discriminator(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot_discriminator, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        labels = ['negation', 'relevancy', 'selection']
        
        if self.learner.learner_config["diffusion_model"] == 'cddpm':
            cgan = load_cgan(layer='nlrl')               
            model = cgan.discriminator
        else:
            gan = load_gan(layer='nlrl')
            model = gan.discriminator
            
        params, init_params = self.extract_parameters(model)
    
        for i, (param, init_param) in enumerate(zip(params, init_params)):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
            ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
            
            ax.set_title(f'{labels[i]} parameters distribution')
            ax.set_xlabel('sigmoid of the learnable weight matrices')
            ax.set_ylabel('number of parameters')
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            figs.append(fig)
            names.append(os.path.join("histogram_plots", "discriminator", f"{labels[i]}"))
            
        return figs, names


class Hist_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        labels = ['negation', 'relevancy', 'selection']
        
        model = load_classifier(layer='nlrl')
        params, init_params = self.extract_parameters(model)
    
        for i, (param, init_param) in enumerate(zip(params, init_params)):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
            ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
            
            ax.set_title(f'{labels[i]} parameters distribution')
            ax.set_xlabel('sigmoid of the learnable weight matrices')
            ax.set_ylabel('number of parameters')
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            figs.append(fig)
            names.append(os.path.join("histogram_plots", "classifier", f"{labels[i]}"))
            
        return figs, names


class Softmax_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Softmax_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating softmax bar plots")

    def consistency_check(self):
        return True
    
    def values(self):
        real_images_list = self.learner.data_storage.get_item("real_images")
        fake_images_list = self.learner.data_storage.get_item("fake_images")
        
        real_images = torch.cat(real_images_list, dim=0)
        fake_images = torch.cat(fake_images_list, dim=0)
        
        selected_real_images = real_images[:50]
        selected_fake_images = fake_images[:50]       
        return selected_real_images, selected_fake_images

    def plot(self):
        names = []
        figs = []
        
        max_images_per_plot = 5 # Define a constant for the maximum number of images per plot
        real_images, fake_images = self.values()
        for types in ["real", "fake"]:
            if types == "real":
                inputs = real_images
            else:
                inputs = fake_images
            sm = torch.nn.Softmax(dim=0)
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the classifier
                classifier = load_classifier(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the classifier
                classifier = load_classifier(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")
                
            model = classifier
            
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            
            # Create subsets for plotting
            subsets = [range(i, min(i + max_images_per_plot, len(inputs))) for i in range(0, len(inputs), max_images_per_plot)]
            
            for subset in subsets:
                fig, axs = plt.subplots(2, len(subset), figsize=(5 * len(subset), 8), squeeze=False)
                
                for idx, image_idx in enumerate(subset):
                    img = (inputs[image_idx].cpu().detach().permute(1, 2, 0)).numpy()
                    pred = preds[image_idx].cpu().detach()
    
                    num_classes = outputs[image_idx].cpu().detach().shape[0]
                    output_softmax = sm(outputs[image_idx].cpu().detach()).numpy()
    
                    axs[0, idx].imshow(img, cmap='gray')
                    axs[0, idx].set_title(f"Predicted: {pred}")
                    axs[0, idx].axis("off")
                    
                    axs[1, idx].bar(range(num_classes), output_softmax)
                    axs[1, idx].set_title("softmax bar plot")
                    axs[1, idx].set_xticks(range(num_classes))
                    axs[1, idx].set_xlabel("class")
                    axs[1, idx].set_ylabel("class probability")
                    axs[1, idx].set_ylim((0, 1))
                    axs[1, idx].set_yticks(torch.arange(0, 1.1, 0.1).tolist())
    
                fig.suptitle(f"{types} images softmax outputs based on classifier")
                dir_name = f"{types}_{subsets.index(subset) + 1}"
                names.append(os.path.join("plots", "analysis_plots", "softmax_plots_classifier", dir_name))
                figs.append(fig)
                plt.close(fig)        
        return figs, names


class Tsne_plot_unet(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot_unet, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating t-SNE plots based on U-Net's layer features")
    
    def consistency_check(self):
        return True

    def get_features(self, noisy_images, time_steps, unet):
        features = []

        def hook(module, input, output):
            features.append(output.detach().cpu())

        handle = unet.conv_out.register_forward_hook(hook)

        # Forward pass through U-Net
        for imgs, t in zip(noisy_images, time_steps):
            _ = unet(imgs, t)

        handle.remove()
        return torch.cat(features)

    def compute_tsne(self, features):
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        
        # Plotting
        figs, names = [], []
        
        for types in ["train", "test"]:
            if types == "train":
                # Process noisy images
                total_noisy_images = self.learner.data_storage.get_item("noise_images")
                batches_per_epoch = int(len(total_noisy_images)/total)
                noisy_images = total_noisy_images[-batches_per_epoch:]
                
                # Process time steps
                total_time_steps = self.learner.data_storage.get_item("time_steps")
                time_steps = total_time_steps[-batches_per_epoch:]
            else:
                # Process noisy images
                total_noisy_images = self.learner.data_storage.get_item("noise_images_test")
                batches_per_epoch = int(len(total_noisy_images)/total)
                noisy_images = total_noisy_images[-batches_per_epoch:]
                
                # Process time steps
                total_time_steps = self.learner.data_storage.get_item("time_steps_test")
                time_steps = total_time_steps[-batches_per_epoch:]
            
            for model in ["initial", "best"]:
                if model == "initial":
                    unet = self.learner._load_initial()
                else:
                    unet = self.learner._load_best()
                
                # Extract features from the discriminator
                features = self.get_features(noisy_images, time_steps, unet)
                # Flatten the features
                features = features.view(features.size(0), -1)
                
                # Compute t-SNE
                tsne_results = self.compute_tsne(features)
                
                # Plotting
                fig, ax = plt.subplots(figsize=(16, 10))
        
                # Plot noisy images
                sns.scatterplot(
                    ax=ax, 
                    x=tsne_results[:, 0], 
                    y=tsne_results[:, 1], 
                    label="Noisy Images", 
                    alpha=0.5, 
                    color='red'
                )
        
                ax.set_title("t-SNE visualization of Noisy Images")
                ax.legend()
        
                # Saving the figure
                figs.append(fig)
                names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_unet", f"{model}_{types}_combined_plot"))
        return figs, names
    
