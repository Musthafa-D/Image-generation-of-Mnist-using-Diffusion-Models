import torch
import os
from ccbdl.utils import DEVICE
from ccbdl.learning.gan import BaseDifussionLearning
from utils import generate_new_images, generate_images
from ccbdl.utils.logging import get_logger
from utils import load_classifier, load_gan
from plots import *


class Learner(BaseDifussionLearning):
    def __init__(self,
                 trial_path: str,
                 model,
                 ddpm,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(train_data, test_data, val_data, trial_path, 
                         learner_config, task=task, logging=logging)

        self.model = model
        self.ddpm = ddpm
        
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
        
        self.plotter.register_default_plot(Loss_plot(self))
        self.plotter.register_default_plot(Image_generation(self))
        self.plotter.register_default_plot(Image_generation_train_test(self))
        self.plotter.register_default_plot(Image_generation_train_test_initial(self))
        # if self.learner_config["diffusion_model"] == "ddpm":
        #     self.plotter.register_default_plot(Attribution_plots_discriminator(self))
        # self.plotter.register_default_plot(Confusion_matrix_gan(self))
        self.plotter.register_default_plot(Fid_plot(self))
        self.plotter.register_default_plot(Psnr_plot(self))
        self.plotter.register_default_plot(Ssim_plot(self))
        #self.plotter.register_default_plot(Noisy_image_generation(self))
        self.plotter.register_default_plot(TimePlot(self))
        # self.plotter.register_default_plot(Tsne_plot_classifier(self))
        # if self.learner_config["diffusion_model"] == "ddpm":
        #     self.plotter.register_default_plot(Attribution_plots_classifier(self))
        
        # if self.learner_config["layer"] == "nlrl":
        #     self.plotter.register_default_plot(Hist_plot_discriminator(self))
        #     self.plotter.register_default_plot(Hist_plot_classifier(self))
        
        # if self.learner_config["diffusion_model"] == "ddpm":            
        #     self.plotter.register_default_plot(Tsne_plot_dis_gan(self))
        #     self.plotter.register_default_plot(Softmax_plot_classifier(self))
        #     self.plotter.register_default_plot(Tsne_plot_unet(self))
        
        # self.classifier = load_classifier(self.learner_config["layer"])
        # self.discriminator = load_gan(self.learner_config["layer"])

        self.parameter_storage.store(self)
        
        #self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters of noise predicting model:")
        #self.parameter_storage.write_tab(self.model.count_learnable_parameters(), "number of learnable parameters of noise predicting model: ")
        
        # Count the total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params}")
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')
    
    def _train_epoch(self, train=True):
        self.logger = get_logger()
        self.logger.info("training.")

        self.model.train()

        for i, data in enumerate(self.train_data):
            self.logger.info("starting train batch")
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            
            num_steps = self.ddpm.n_steps
            
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            self.logger.info("creating noise tensors and time steps")
            noise_tensor = torch.randn_like(images).to(self.device) 
            time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)

            # Computing the noisy images based on images and the time-steps (forward process)
            self.logger.info("getting the noisy images")
            noisy_images = self.noising_images(images, time_steps, noise_tensor)
            
            # Getting ddpm estimation of noise based on the images and the time-step
            self.logger.info("predicitng the noise")
            noise_pred = self.noise_prediction(noisy_images, time_steps)
            
            self.logger.info("train loss")

            self.train_loss = self.criterion(noise_pred, noise_tensor)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()

            self.data_storage.store([self.epoch, self.batch, self.train_loss, self.test_loss])

            if train:
                self.batch += 1
                if self.epoch == 0:
                    if i == 0:
                        self.original_images_initial = images
                        
                        fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"])
                        generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=1, height=32, width=32, 
                                                                denoising_option=self.learner_config["denoising_option"], 
                                                                fixed_noise=noise_tensor)
                        self.fake_images_initial = fake_images.detach()
                        self.generated_images_initial = generated_images
                        self.noise_tensors_initial = noise_tensor
                        self.noises_initial = noisy_images
                        self.labels_initial = labels
                        
                if i == len(self.train_data) - 2:
                    self.data_storage.dump_store("real_images", images)
                    
                    fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"])
                    generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=1, height=32, width=32, 
                                                            denoising_option=self.learner_config["denoising_option"], 
                                                            fixed_noise=noise_tensor)
                    self.data_storage.dump_store("fake_images", fake_images.detach())  
                    self.data_storage.dump_store("generated_images", generated_images)
                    self.data_storage.dump_store("noise_tensors", noise_tensor)
                    self.data_storage.dump_store("noise_images", noisy_images)
                    self.data_storage.dump_store("labels", labels)
                    self.data_storage.dump_store("time_steps", time_steps)
    
    def _test_epoch(self):
        self.logger = get_logger()
        self.logger.info("testing.")
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict()},
                       self.initial_save_path)
    
        self.model.eval()
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                self.logger.info("starting test batch")
    
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device).long()
    
                num_steps = self.ddpm.n_steps
    
                # Creating noise tensors and time steps for the test data
                self.logger.info("creating noise tensors and time steps for test data")
                noise_tensor = torch.randn_like(images).to(self.device)
                time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)
    
                # Getting the noisy images for the test data
                self.logger.info("getting the noisy images for test data")
                noisy_images = self.noising_images(images, time_steps, noise_tensor)
    
                # Denoising the noisy images using the ddpm
                self.logger.info("predicitng the noise for the test data")
                noise_pred = self.noise_prediction(noisy_images, time_steps)
    
                # Calculate the loss for the test data
                self.logger.info("calculating test loss")
                loss += self.criterion(noise_pred, noise_tensor)
                
                if self.epoch == 0:
                    if i == 0:
                        self.original_images_test_initial = images
                        
                        fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"])
                        generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=1, height=32, width=32, 
                                                                denoising_option=self.learner_config["denoising_option"], 
                                                                fixed_noise=noise_tensor)
                        self.fake_images_test_initial = fake_images.detach()
                        self.generated_images_test_initial = generated_images
                        self.noise_tensors_test_initial = noise_tensor
                        self.noises_test_initial = noisy_images
                        self.labels_test_initial = labels
                
                if i == len(self.test_data) - 2:
                    self.data_storage.dump_store("real_images_test", images)
                    
                    fake_images = generate_images(self.ddpm, self.device, noisy_images, 
                                                  self.learner_config["denoising_option"])
                    generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=1, height=32, width=32, 
                                                            denoising_option=self.learner_config["denoising_option"], 
                                                            fixed_noise=noise_tensor)
                    self.data_storage.dump_store("fake_images_test", fake_images.detach())        
                    self.data_storage.dump_store("generated_images_test", generated_images)
                    self.data_storage.dump_store("noise_tensors_test", noise_tensor)
                    self.data_storage.dump_store("noise_images_test", noisy_images)
                    self.data_storage.dump_store("labels_test", labels)
                    self.data_storage.dump_store("time_steps_test", time_steps)
        
        self.test_loss = loss / (i+1)
    
    def _validate_epoch(self):
        pass

    def noising_images(self, ins, time_steps, noise_tensor):
        return self.ddpm.noising_images(ins, time_steps, noise_tensor)

    def noise_prediction(self, x, t):
        return self.ddpm.noise_prediction(x, t)
    
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
            
            self.noise = torch.randn(20, 1, 32, 32).to(self.device)
        
        generated_images_fixed, noise_fixed = generate_new_images(self.ddpm, n_samples=20, device=self.device,
                                                channels=1, height=32, width=32, 
                                                denoising_option=self.learner_config["denoising_option"], 
                                                fixed_noise=self.noise)
        
        generated_images, noise_random = generate_new_images(self.ddpm, n_samples=20, device=self.device,
                                                channels=1, height=32, width=32, 
                                                denoising_option=self.learner_config["denoising_option"])
        
        self.data_storage.dump_store("generated_images_random", generated_images)
        self.data_storage.dump_store("random_noise", noise_random)
        
        self.data_storage.dump_store("generated_images_fixed", generated_images_fixed)
        self.data_storage.dump_store("fixed_noise", noise_fixed)
        
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
        
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model


class Conditional_Learner(Learner):
    def __init__(self,
                 trial_path: str,
                 model,
                 ddpm,
                 train_data,
                 test_data,
                 val_data,
                 task,
                 learner_config: dict,
                 logging):

        super().__init__(trial_path, model, ddpm, train_data, test_data, val_data, task, learner_config, logging)
        
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
            
            num_steps = self.ddpm.n_steps
            
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            self.logger.info("creating noise tensors and time steps")
            noise_tensor = torch.randn_like(images).to(self.device) 
            time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)

            # Computing the noisy images based on images and the time-steps (forward process)
            self.logger.info("getting the noisy images")
            noisy_images = self.noising_images(images, time_steps, noise_tensor)
            
            # Getting ddpm estimation of noise based on the images and the time-step
            self.logger.info("denoising the noisy images")
            noise_pred = self.noise_prediction(noisy_images, time_steps, labels)
            
            self.logger.info("train loss")

            self.train_loss = self.criterion(noise_pred, noise_tensor)
            
            self.logger.info("updating network")

            self.train_loss.backward()
            self.optimizer.step()

            self.data_storage.store([self.epoch, self.batch, self.train_loss, self.test_loss])

            if train:
                self.batch += 1
                if self.epoch == 0:
                    if i == 0:
                        self.original_images_initial = images
                        
                        fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"],
                                                      labels)
                        generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=1, height=32, width=32, 
                                                                denoising_option=self.learner_config["denoising_option"],
                                                                labels=labels,
                                                                fixed_noise=noise_tensor)
                        self.fake_images_initial = fake_images.detach()
                        self.generated_images_initial = generated_images
                        self.noise_tensors_initial = noise_tensor
                        self.noises_initial = noisy_images
                        self.labels_initial = labels
                        
                if i == len(self.train_data) - 2:
                    self.data_storage.dump_store("real_images", images)
                    
                    fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"],
                                                  labels)
                    generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=1, height=32, width=32, 
                                                            denoising_option=self.learner_config["denoising_option"],
                                                            labels=labels,
                                                            fixed_noise=noise_tensor)
                    self.data_storage.dump_store("fake_images", fake_images.detach())  
                    self.data_storage.dump_store("generated_images", generated_images)
                    self.data_storage.dump_store("noise_tensors", noise_tensor)
                    self.data_storage.dump_store("noise_images", noisy_images)
                    self.data_storage.dump_store("labels", labels)
    
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
    
                num_steps = self.ddpm.n_steps
    
                # Creating noise tensors and time steps for the test data
                self.logger.info("creating noise tensors and time steps for test data")
                noise_tensor = torch.randn_like(images).to(self.device)
                time_steps = torch.randint(0, num_steps, (len(images),)).to(self.device)
    
                # Getting the noisy images for the test data
                self.logger.info("getting the noisy images for test data")
                noisy_images = self.noising_images(images, time_steps, noise_tensor)
    
                # Denoising the noisy images using the ddpm
                self.logger.info("denoising the noisy test images")
                noise_pred = self.noise_prediction(noisy_images, time_steps, labels)
    
                # Calculate the loss for the test data
                self.logger.info("calculating test loss")
                loss += self.criterion(noise_pred, noise_tensor)
                
                if self.epoch == 0:
                    if i == 0:
                        self.original_images_test_initial = images
                        
                        fake_images = generate_images(self.ddpm, self.device, noisy_images, self.learner_config["denoising_option"],
                                                      labels)
                        generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                                channels=1, height=32, width=32, 
                                                                denoising_option=self.learner_config["denoising_option"], 
                                                                labels=labels,
                                                                fixed_noise=noise_tensor)
                        self.fake_images_test_initial = fake_images.detach()
                        self.generated_images_test_initial = generated_images
                        self.noise_tensors_test_initial = noise_tensor
                        self.noises_test_initial = noisy_images
                        self.labels_test_initial = labels
                
                if i == len(self.test_data) - 2:
                    self.data_storage.dump_store("real_images_test", images)
                    
                    fake_images = generate_images(self.ddpm, self.device, noisy_images, 
                                                  self.learner_config["denoising_option"], labels)
                    generated_images, _ = generate_new_images(self.ddpm, n_samples=len(images), device=self.device,
                                                            channels=1, height=32, width=32, 
                                                            denoising_option=self.learner_config["denoising_option"],
                                                            labels=labels,
                                                            fixed_noise=noise_tensor)
                    self.data_storage.dump_store("fake_images_test", fake_images.detach())        
                    self.data_storage.dump_store("generated_images_test", generated_images)
                    self.data_storage.dump_store("noise_tensors_test", noise_tensor)
                    self.data_storage.dump_store("noise_images_test", noisy_images)
                    self.data_storage.dump_store("labels_test", labels)
        
        self.test_loss = loss / (i+1)
    
    def noising_images(self, ins, time_steps, noise_tensor):
        return self.ddpm.noising_images(ins, time_steps, noise_tensor)

    def noise_prediction(self, x, t, c):
        return self.ddpm.noise_prediction_labels(x, t, c)

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
            
            self.noise = torch.randn(20, 1, 32, 32).to(self.device)
            self.test_labels = torch.tensor([i % 10 for i in range(20)], device=self.device)
        
        generated_images_fixed, noise_fixed = generate_new_images(self.ddpm, n_samples=20, device=self.device,
                                                channels=1, height=32, width=32, 
                                                denoising_option=self.learner_config["denoising_option"], 
                                                fixed_noise=self.noise,
                                                labels=self.test_labels)
        
        generated_images, noise_random = generate_new_images(self.ddpm, n_samples=20, device=self.device,
                                                channels=1, height=32, width=32, 
                                                denoising_option=self.learner_config["denoising_option"],
                                                labels=self.test_labels)

        self.data_storage.dump_store("generated_images_random", generated_images)
        self.data_storage.dump_store("random_noise", noise_random)
        
        self.data_storage.dump_store("generated_images_fixed", generated_images_fixed)
        self.data_storage.dump_store("fixed_noise", noise_fixed)
        
        self.data_storage.dump_store("gen_labels", self.test_labels)
        
        self.data_storage.dump_store("epochs_gen", self.epoch)
