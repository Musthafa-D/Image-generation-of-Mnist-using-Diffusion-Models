import torch
from ccbdl.utils import DEVICE
from utils import generate_images_latent, generate_new_images_latent
from ccbdl.utils.logging import get_logger
from plots import *
import sys
from ccbdl.learning.gan import BaseDifussionLearning


class Learner_Latent(BaseDifussionLearning):
    def __init__(self,
                 trial_path: str,
                 model,
                 latent_dm,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(train_data, test_data, val_data, trial_path, 
                         learner_config, task=task, logging=logging)

        self.model = model
        self.latent_dm = latent_dm
        
        self.figure_storage.dpi=100
        
        self.device = DEVICE
        print(self.device)
        
        self.criterion_name = learner_config["criterion"]
        self.lr_exp = learner_config["learning_rate_exp"]
        self.learner_config = learner_config
        
        self.lr = 10**self.lr_exp

        self.criterion = getattr(torch.nn, self.criterion_name)(reduction='mean').to(self.device)
        
        self.optimizer = getattr(torch.optim, self.optimizer)(self.model.parameters(), lr=self.lr)

        self.train_data = train_data
        self.test_data = test_data

        self.result_folder = trial_path
        
        self.plotter.register_default_plot(TimePlot(self))
        self.plotter.register_default_plot(Loss_plot(self))
        self.plotter.register_default_plot(Image_generation(self))
        self.plotter.register_default_plot(Image_generation_train_test(self))
        # self.plotter.register_default_plot(Attribution_plots_discriminator(self))
        # self.plotter.register_default_plot(Confusion_matrix_gan(self))
        self.plotter.register_default_plot(Fid_plot(self))
        self.plotter.register_default_plot(Psnr_plot(self))
        self.plotter.register_default_plot(Ssim_plot(self))
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        # self.plotter.register_default_plot(Attribution_plots_classifier(self))
        
        # if self.learner_config["layer"] == "nlrl":
        #     self.plotter.register_default_plot(Hist_plot_discriminator(self))
        #     self.plotter.register_default_plot(Hist_plot_classifier(self))
        
        # if self.learner_config["diffusion_model"] == "latent_dm":
        #     self.plotter.register_default_plot(Tsne_plot_dis_gan(self))

        self.parameter_storage.store(self)
        
        #self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters of latent noise predicting model:")
        #self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters of latent noise predicting model: ")
        
        # Count the total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
    
    def _train_epoch(self, train=True):
        self.logger = get_logger()
        self.logger.info("training.")

        self.model.train()

        for i, data in enumerate(self.train_data):
            self.logger.info("starting train batch")
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            
            num_steps = self.latent_dm.n_steps
            
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            self.logger.info("creating noise tensors and time steps") 
            time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)

            # Computing the noisy images based on images and the time-steps (forward process)
            self.logger.info("getting the noisy images")
            noisy_images_latent, noise_tensor_latent, images_latent = self.noising_images(images, time_steps)
            
            # Getting latent_dm estimation of noise based on the images and the time-step
            self.logger.info("predicting the noise for the latent")
            noise_pred_de, noise_pred_latent = self.noise_prediction(noisy_images_latent, 
                                                            time_steps)
            
            self.logger.info("train loss")

            self.train_loss = self.criterion(noise_pred_latent, noise_tensor_latent)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()

            self.data_storage.store([self.epoch, self.batch, self.train_loss, self.test_loss])

            if train:
                self.batch += 1
                        
                if i == len(self.train_data) - 2:
                    self.data_storage.dump_store("real_images", images)
                    self.data_storage.dump_store("real_images_latent", images_latent)
                    
                    fake_images_de = generate_images_latent(self.latent_dm, self.device, 
                                                                                    noisy_images_latent, 
                                                                                    self.learner_config["denoising_option"])
                    generated_images_de, _ = generate_new_images_latent(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=64, height=4, width=4, 
                                                                denoising_option=self.learner_config["denoising_option"],
                                                                fixed_noise=noise_tensor_latent)
                    self.data_storage.dump_store("fake_images", fake_images_de.detach())  
                    self.data_storage.dump_store("generated_images", generated_images_de.detach())
                    self.data_storage.dump_store("noise_tensors_latent", noise_tensor_latent)
                    self.data_storage.dump_store("noisy_images_latent", noisy_images_latent)
                    self.data_storage.dump_store("labels", labels)
                    self.data_storage.dump_store("time_steps", time_steps)
    
    def _test_epoch(self):
        self.logger = get_logger()
        self.logger.info("testing.")
    
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                self.logger.info("starting test batch")
    
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).long()
    
                num_steps = self.latent_dm.n_steps
    
                # Creating noise tensors and time steps for the test data
                self.logger.info("creating noise tensors and time steps for test data")
                time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)
    
                # Getting the noisy images for the test data
                self.logger.info("getting the noisy images for test data")
                noisy_images_latent, noise_tensor_latent, images_latent = self.noising_images(images, time_steps)
    
                # Denoising the noisy images using the latent_dm
                self.logger.info("predicting the noise for the latent for the test data")
                noise_pred_de, noise_pred_latent = self.noise_prediction(noisy_images_latent, 
                                                                time_steps)
    
                # Calculate the loss for the test data
                self.logger.info("calculating test loss")
                loss += self.criterion(noise_pred_latent, noise_tensor_latent)
                
                
                
                if i == len(self.test_data) - 2:
                    self.data_storage.dump_store("real_images_test", images)
                    self.data_storage.dump_store("real_images_test_latent", images_latent)
                    
                    fake_images_de = generate_images_latent(self.latent_dm, self.device, 
                                                                                    noisy_images_latent, 
                                                                                    self.learner_config["denoising_option"])
                    generated_images_de, _ = generate_new_images_latent(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=64, height=4, width=4, 
                                                            denoising_option=self.learner_config["denoising_option"], 
                                                            fixed_noise=noise_tensor_latent)
                    self.data_storage.dump_store("generated_images_test", generated_images_de.detach())
                    self.data_storage.dump_store("fake_images_test", fake_images_de.detach())
                    self.data_storage.dump_store("noisy_images_test_latent", noisy_images_latent)
                    self.data_storage.dump_store("noise_tensors_test_latent", noise_tensor_latent)
                    self.data_storage.dump_store("labels_test", labels)
        
        self.test_loss = loss / (i+1)
    
    def _validate_epoch(self):
        pass

    def noising_images(self, ins, time_steps):
        return self.latent_dm.noising_images(ins, time_steps)

    def noise_prediction(self, x, t):
        return self.latent_dm.noise_prediction(x, t)
    
    def _update_best(self):
        if self.test_loss < self.best_values["TestLoss"]:
            self.best_values["TrainLoss"] = self.train_loss.item()
            self.best_values["TestLoss"] = self.test_loss.item()
            self.best_values["Batch"] = self.batch

            self.best_state_dict = self.model.state_dict()

    def evaluate(self):
        self.end_values = {"TrainLoss":      self.train_loss.item(),
                           "TestLoss":       self.test_loss.item(),
                           "Batch":          self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"TrainLoss":      self.train_loss.item(),
                               "TestLoss":       self.test_loss.item(),
                               "Batch":          self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'TrainLoss': self.train_loss.item(),
                                    "TestLoss":  self.test_loss.item(),
                                    'model_state_dict': self.model.state_dict()}
            
            self.noise = torch.randn(10, self.latent_dm.in_channels_de, 4, 4).to(self.device)
        
        gen_imgs_de_fixed, noise_latent_fixed = generate_new_images_latent(self.latent_dm, n_samples=10, 
                                                device=self.device,
                                                channels=self.latent_dm.in_channels_de, 
                                                height=4, width=4,
                                                denoising_option=self.learner_config["denoising_option"],
                                                fixed_noise=self.noise)
        
        gen_imgs_de, noise_latent = generate_new_images_latent(self.latent_dm, n_samples=10, 
                                                device=self.device,
                                                channels=self.latent_dm.in_channels_de, 
                                                height=4, width=4,
                                                denoising_option=self.learner_config["denoising_option"])
        
        self.data_storage.dump_store("random_noise_latent", noise_latent)
        self.data_storage.dump_store("generated_images", gen_imgs_de)
        
        self.data_storage.dump_store("fixed_noise_latent", noise_latent_fixed)
        self.data_storage.dump_store("generated_images_fixed", gen_imgs_de_fixed)
        
        self.data_storage.dump_store("epochs_gen", self.epoch)
            
    def _save(self):
        torch.save(self.init_state_dict, self.init_save_path)
        
        torch.save({'epoch': self.epoch,
                    'best_values': self.best_values,
                    'model_state_dict': self.best_state_dict},
                   self.best_save_path)

        torch.save({'epoch': self.epoch,
                    'batch': self.batch,
                    'TrainLoss': self.train_loss.item(),
                    "TestLoss":  self.test_loss.item(),
                    'model_state_dict': self.model.state_dict()},
                   self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")


class Conditional_Learner_Latent(Learner_Latent):
    def __init__(self,
                 trial_path: str,
                 model,
                 latent_dm,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(trial_path, model, latent_dm, train_data, test_data, val_data, task, learner_config, logging)
        
        # self.plotter.register_default_plot(Tsne_plot_dis_cgan(self))
        # self.plotter.register_default_plot(Tsne_plot_images_separate(self))
        # self.plotter.register_default_plot(Tsne_plot_images_combined(self))
        # self.plotter.register_default_plot(Confusion_matrix_classifier(self, **{"ticks": torch.arange(0, 10, 1).numpy()}))
    
    def _train_epoch(self, train=True):
        self.logger = get_logger()
        self.logger.info("training.")

        self.model.train()

        for i, data in enumerate(self.train_data):
            self.logger.info("starting train batch")
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            
            num_steps = self.latent_dm.n_steps
            
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            self.logger.info("creating noise tensors and time steps") 
            time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)

            # Computing the noisy images based on images and the time-steps (forward process)
            self.logger.info("getting the noisy images")
            noisy_images_latent, noise_tensor_latent, images_latent = self.noising_images(images, time_steps, labels.reshape(len(images), -1))
            
            # Getting latent_dm estimation of noise based on the images and the time-step
            self.logger.info("predicting the noise for the latent")
            noise_pred_de, noise_pred_latent = self.noise_prediction(noisy_images_latent, 
                                                            time_steps,
                                                            labels)
            
            self.logger.info("train loss")

            self.train_loss = self.criterion(noise_pred_latent, noise_tensor_latent)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()

            self.data_storage.store([self.epoch, self.batch, self.train_loss, self.test_loss])

            if train:
                self.batch += 1
                
                if i == len(self.train_data) - 2:
                    self.data_storage.dump_store("real_images", images)
                    self.data_storage.dump_store("real_images_latent", images_latent)
                    
                    fake_images_de = generate_images_latent(self.latent_dm, self.device, 
                                                                                    noisy_images_latent, 
                                                                                    self.learner_config["denoising_option"],
                                                                                    labels)
                    generated_images_de, _ = generate_new_images_latent(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=64, height=4, width=4, 
                                                                denoising_option=self.learner_config["denoising_option"], 
                                                                labels=labels,
                                                                fixed_noise=noise_tensor_latent)
                    self.data_storage.dump_store("fake_images", fake_images_de.detach())  
                    self.data_storage.dump_store("generated_images", generated_images_de.detach())
                    self.data_storage.dump_store("noise_tensors_latent", noise_tensor_latent)
                    self.data_storage.dump_store("noisy_images_latent", noisy_images_latent)
                    self.data_storage.dump_store("labels", labels)
                    self.data_storage.dump_store("time_steps", time_steps)
    
    def _test_epoch(self):
        self.logger = get_logger()
        self.logger.info("testing.")
    
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                self.logger.info("starting test batch")
    
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).long()
    
                num_steps = self.latent_dm.n_steps
    
                # Creating noise tensors and time steps for the test data
                self.logger.info("creating noise tensors and time steps for test data")
                time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)
    
                # Getting the noisy images for the test data
                self.logger.info("getting the noisy images for test data")
                noisy_images_latent, noise_tensor_latent, images_latent = self.noising_images(images, time_steps, labels.reshape(len(images), -1))
    
                # Denoising the noisy images using the latent_dm
                self.logger.info("predicting the noise for the latent for the test data")
                noise_pred_de, noise_pred_latent = self.noise_prediction(noisy_images_latent, 
                                                                time_steps,
                                                                labels)
    
                # Calculate the loss for the test data
                self.logger.info("calculating test loss")
                loss += self.criterion(noise_pred_latent, noise_tensor_latent)
                
                if i == len(self.test_data) - 2:
                    self.data_storage.dump_store("real_images_test", images)
                    self.data_storage.dump_store("real_images_test_latent", images_latent)
                    
                    fake_images_de = generate_images_latent(self.latent_dm, self.device, 
                                                                                    noisy_images_latent, 
                                                                                    self.learner_config["denoising_option"],
                                                                                    labels)
                    generated_images_de, _ = generate_new_images_latent(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=64, height=4, width=4, 
                                                            denoising_option=self.learner_config["denoising_option"], 
                                                            labels=labels,
                                                            fixed_noise=noise_tensor_latent)
                    self.data_storage.dump_store("generated_images_test", generated_images_de.detach())
                    self.data_storage.dump_store("fake_images_test", fake_images_de.detach())
                    self.data_storage.dump_store("noisy_images_test_latent", noisy_images_latent)
                    self.data_storage.dump_store("noise_tensors_test_latent", noise_tensor_latent)
                    self.data_storage.dump_store("labels_test", labels)
        
        self.test_loss = loss / (i+1)
    
    def noising_images(self, ins, time_steps, labels):
        return self.latent_dm.noising_images_labels(ins, time_steps, labels)

    def noise_prediction(self, x, t, c):
        return self.latent_dm.noise_prediction_labels(x, t, c)

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"TrainLoss":      self.train_loss.item(),
                               "TestLoss":       self.test_loss.item(),
                               "Batch":          self.batch}

            self.init_state_dict = {'epoch': self.epoch,
                                    'batch': self.batch,
                                    'TrainLoss': self.train_loss.item(),
                                    "TestLoss":  self.test_loss.item(),
                                    'model_state_dict': self.model.state_dict()}
            
            self.test_labels = torch.tensor([i % 10 for i in range(10)], device=self.device)
            self.noise = torch.randn(10, self.latent_dm.in_channels_de, 4, 4).to(self.device)
        
        gen_imgs_de_fixed, noise_latent_fixed = generate_new_images_latent(self.latent_dm, n_samples=10, 
                                                device=self.device,
                                                channels=self.latent_dm.in_channels_de, 
                                                height=4, width=4, 
                                                fixed_noise=self.noise,
                                                denoising_option=self.learner_config["denoising_option"],
                                                labels=self.test_labels)
        
        gen_imgs_de, noise_latent = generate_new_images_latent(self.latent_dm, n_samples=10, 
                                                device=self.device,
                                                channels=self.latent_dm.in_channels_de, 
                                                height=4, width=4,
                                                denoising_option=self.learner_config["denoising_option"],
                                                labels=self.test_labels)
        
        self.data_storage.dump_store("random_noise_latent", noise_latent)
        self.data_storage.dump_store("generated_images", gen_imgs_de)
        
        self.data_storage.dump_store("fixed_noise_latent", noise_latent_fixed)
        self.data_storage.dump_store("generated_images_fixed", gen_imgs_de_fixed)
        
        self.data_storage.dump_store("gen_labels", self.test_labels)
        self.data_storage.dump_store("epochs_gen", self.epoch)
        
