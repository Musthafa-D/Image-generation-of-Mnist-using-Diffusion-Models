from learner import Learner, Conditional_Learner
from learner_latent import Learner_Latent, Conditional_Learner_Latent
from ccbdl.parameter_optimizer.optuna_base import BaseOptunaParamOptimizer
from ccbdl.utils import DEVICE
from datetime import datetime, timedelta
from data_loader import prepare_data
from Networks.unet import Conditional_UNet#,  UNet
from Networks.unet_new import UNet, UNet_conditional
from Networks.unet_latent import UNet_Latent, Conditional_UNet_Latent
from Networks.encoder_decoder import Encoder_Decoder, Conditional_Encoder_Decoder
from Networks.ddpm import DDPM
from Networks.latent_diffusion import Latent_Diffusion
import optuna
import os
import ccbdl
import matplotlib.pyplot as plt


class Optuna(BaseOptunaParamOptimizer):
    def __init__(self,
                 study_config: dict,
                 optimize_config: dict,
                 network_config: dict,
                 data_config: dict,
                 learner_config: dict,
                 diffusion_config: dict,
                 config,
                 study_path: str,
                 comment: str = "",
                 config_path: str = "",
                 debug: bool = False,
                 logging: bool = False):
        # get sampler and pruner for parent class
        if "sampler" in study_config.keys():
            if hasattr(optuna.samplers, study_config["sampler"]["name"]):
                sampler = getattr(
                    optuna.samplers, study_config["sampler"]["name"])()
        else:
            sampler = optuna.samplers.TPESampler()

        if "pruner" in study_config.keys():
            if hasattr(optuna.pruners, study_config["pruner"]["name"]):
                pruner = getattr(
                    optuna.pruners, study_config["pruner"]["name"])()
        else:
            pruner = None

        super().__init__(study_config["direction"], study_config["study_name"], study_path,
                         study_config["number_of_trials"], data_config["task"], comment, 
                         study_config["optimization_target"],
                         sampler, pruner, config_path, debug, logging)

        self.optimize_config = optimize_config
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.study_config = study_config
        self.diffusion_config = diffusion_config
        
        self.result_folder = study_path
        self.config = config

        optuna.logging.disable_default_handler()
        self.create_study()

        self.create_study()
        self.study_name = study_config["study_name"]
        self.durations = []
        
        self.learnable_parameters_list = []
        self.optimization_target_values = []
        self.lr_list = []

    def _objective(self, trial):
        start_time = datetime.now()

        print("\n\n******* Trial " + str(trial.number) +
              " has started" + "*******\n")
        trial_folder = f'trial_{trial.number}'
        trial_path = os.path.join(self.result_folder, trial_folder)
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        # suggest parameters
        suggested = self._suggest_parameters(self.optimize_config, trial)

        self.learner_config["learning_rate_exp"] = suggested["learning_rate_exp"]
        self.learner_config["denoising_option"] = suggested["denoising_option"]

        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)
        
        if self.learner_config["diffusion_model"] == "ddpm":
            if self.learner_config["noise_prediction_model"] == "unet":
                # model = UNet(self.learner_config["n_steps"], **self.network_config["unet"]).to(DEVICE)
                model = UNet(c_in=1, c_out=1, time_dim=256, device="cuda").to(DEVICE)
                
            elif self.learner_config["noise_prediction_model"] == "en_de":
                model = Encoder_Decoder(self.learner_config["n_steps"], **self.network_config["en_de"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be unet, or en_de only.")
                
            ddpm = DDPM(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["ddpm"])
                
            self.learner = Learner(trial_path,
                                   model,
                                   ddpm,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
            
        elif self.learner_config["diffusion_model"] == "cddpm":
            if self.learner_config["noise_prediction_model"] == "conditional_unet":
                # model = Conditional_UNet(self.learner_config["n_steps"], **self.network_config["conditional_unet"]).to(DEVICE)
                model = UNet_conditional(c_in=1, c_out=1, time_dim=256, num_classes=10, device="cuda").to(DEVICE)
                
            elif self.learner_config["noise_prediction_model"] == "conditional_en_de":
                model = Conditional_Encoder_Decoder(self.learner_config["n_steps"], **self.network_config["conditional_en_de"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be conditional_unet, or conditional_en_de only.")
                
            ddpm = DDPM(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["ddpm"])
                
            self.learner = Conditional_Learner(trial_path,
                                   model,
                                   ddpm,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
        
        elif self.learner_config["diffusion_model"] == "latent_dm":
            if self.learner_config["noise_prediction_model"] == "unet_latent":
                model = UNet_Latent(self.learner_config["n_steps"], **self.network_config["unet_latent"]).to(DEVICE)
                
            elif self.learner_config["noise_prediction_model"] == "en_de_latent":
                model = Encoder_Decoder(self.learner_config["n_steps"], **self.network_config["en_de_latent"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be unet_latent, or en_de_latent only.")
                
            latent_dm = Latent_Diffusion(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["latent"])
                
            self.learner = Learner_Latent(trial_path,
                                   model,
                                   latent_dm,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
        
        elif self.learner_config["diffusion_model"] == "conditional_latent_dm":
            if self.learner_config["noise_prediction_model"] == "conditional_unet_latent":
                model = Conditional_UNet_Latent(self.learner_config["n_steps"], **self.network_config["conditional_unet_latent"]).to(DEVICE)
                
            elif self.learner_config["noise_prediction_model"] == "conditional_en_de_latent":
                model = Encoder_Decoder(self.learner_config["n_steps"], **self.network_config["conditional_en_de_latent"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be conditional_unet_latent, or conditional_en_de_latent only.")
                
            latent_dm = Latent_Diffusion(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["latent"])
                
            self.learner = Conditional_Learner_Latent(trial_path,
                                   model,
                                   latent_dm,
                                   train_data,
                                   test_data,
                                   val_data,
                                   self.task,
                                   self.learner_config,
                                   logging=True)
        
        else:
            raise ValueError(
                f"Invalid value for model: {self.learner_config['model']}, it should be ddpm, cddpm, latent_dm or conditional_latent_dm only.")

        self.learner.parameter_storage.write("Current config:-\n")
        self.learner.parameter_storage.store(self.config)
        
        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write(
            f"Start Time of diffusion training and evaluation in this Trial {trial.number}: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.store(suggested, header="suggested_parameters")
        self.learner.parameter_storage.write("\n")

        print(f"\n\n******* Trial {trial.number} is completed*******")

        end_time = datetime.now()

        self.learner.parameter_storage.write(
            f"End Time of diffusion training and evaluation in this Trial {trial.number}: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations.append(self.duration_trial)

        self.learner.parameter_storage.write(
            f"Duration of diffusion training and evaluation in this Trial {trial.number}: {str(self.duration_trial)[:-7]}\n")
        
        # Storing the number of learnable parameters for this trial
        learnable_params = self.learner.model.count_learnable_parameters()
        self.learnable_parameters_list.append(learnable_params)
        
        # Storing the learning rate for this trial
        lr_params = self.learner.lr
        self.lr_list.append(lr_params)
        
        # Storing the optimization target value for this trial
        optim_target_val = self.learner.best_values[self.optimization_target]
        self.optimization_target_values.append(optim_target_val)


        return self.learner.best_values[self.optimization_target]

    def start_study(self):
        self.study.optimize(self._objective, n_trials=self.number_of_trials,)

    def eval_study(self):
        if self.logging:
            self.logger.info("evaluating study")   
        start_time = datetime.now()
        
        parameter_storage = ccbdl.storages.storages.ParameterStorage(
            self.result_folder, file_name="study_info.txt")

        parameter_storage.write("******* Summary " +
                                "of " + self.study_name + " *******")
        pruned_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if self.logging:
            self.logger.info("creating optuna plots")
        
        sub_folder = os.path.join(self.result_folder, 'study_plots', 'optuna_plots')
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        self.figure_storage = ccbdl.storages.storages.FigureStorage(
            sub_folder, types=("png", "pdf"))

        figures_list = []
        figures_names = []
    
        fig = optuna.visualization.plot_optimization_history(self.study)
        figures_list.append(fig)
        figures_names.append("optimization_history")
        
        fig = optuna.visualization.plot_contour(
            self.study, params=["learning_rate_exp", "denoising_option", "num_layers", "block_layers"])
        figures_list.append(fig)
        figures_names.append("contour")
    
        fig = optuna.visualization.plot_parallel_coordinate(
            self.study, params=["learning_rate_exp", "denoising_option", "num_layers", "block_layers"])
        figures_list.append(fig)
        figures_names.append("parallel_coordinate")
    
        fig = optuna.visualization.plot_param_importances(self.study)
        figures_list.append(fig)
        figures_names.append("param_importances")
    
        fig = optuna.visualization.plot_slice(
            self.study, params=["learning_rate_exp", "denoising_option", "num_layers", "block_layers"])
        figures_list.append(fig)
        figures_names.append("plot_slice")
    
        # Now use store_multi to store all figures at once
        self.figure_storage.store_multi(figures_list, figures_names)
        
        if self.logging:
            self.logger.info("creating loss vs parameters and learning plot")
            
        param_folder = os.path.join(self.result_folder, 'study_plots', 'loss_plots')
        if not os.path.exists(param_folder):
            os.makedirs(param_folder)
        self.fig_storage = ccbdl.storages.storages.FigureStorage(
            param_folder, types=("png", "pdf"))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(self.learnable_parameters_list, self.optimization_target_values)
        ax.set_xlabel("number of learnable parameters")
        ax.set_ylabel("test loss")
        ax.set_title("test loss vs learnable Parameter")
        ax.grid(True)
        ax.set_ylim([0, 0.2])
        self.fig_storage.store(fig, "param_loss_plot")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(self.lr_list, self.optimization_target_values)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("test loss")
        ax.set_title("test loss vs learning rate")
        ax.grid(True)
        ax.set_ylim([0, 0.2])
        self.fig_storage.store(fig, "lr_loss_plot")

        
        end_time = datetime.now()
        self.overall_duration = sum(self.durations, timedelta()) + (end_time - start_time)
        
        parameter_storage.write("\nStudy statistics: ")
        parameter_storage.write(
            f"  Number of finished trials: {len(self.study.trials)}")
        parameter_storage.write(
            f"  Number of pruned trials: {len(pruned_trials)}")
        parameter_storage.write(
            f"  Number of complete trials: {len(complete_trials)}")
        parameter_storage.write(
            f"  Time of this entire study: {str(self.overall_duration)[:-7]}")
        parameter_storage.write(
            f"\nBest trial: Nr {self.study.best_trial.number}")
        parameter_storage.write(f"  Best Value: {self.study.best_trial.value}")

        parameter_storage.write("  Params: ")
        for key, value in self.study.best_trial.params.items():
            parameter_storage.write(f"    {key}: {value}")
        parameter_storage.write("\n")      
