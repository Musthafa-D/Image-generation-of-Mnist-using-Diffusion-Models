from learner import Learner, Conditional_Learner
from learner_latent import Learner_Latent, Conditional_Learner_Latent
from ccbdl.utils import DEVICE
from datetime import datetime
from data_loader import prepare_data
from Networks.unet_new import UNet, UNet_conditional
from Networks.unet_latent import UNet_Latent, UNet_conditional_Latent
from Networks.ddpm import DDPM
from Networks.latent_diffusion import Latent_Diffusion


class Normal_run:
    def __init__(self,
                 task,
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
        
        self.network_config = network_config
        self.data_config = data_config
        self.learner_config = learner_config
        self.result_folder = study_path
        self.config = config
        self.task = task
        self.diffusion_config = diffusion_config
    
    def execute(self):
        start_time = datetime.now()
        
        print("\n\n******* Run is started*******")
        
        # get data
        train_data, test_data, val_data = prepare_data(self.data_config)
        
        if self.learner_config["diffusion_model"] == "ddpm":
            if self.learner_config["noise_prediction_model"] == "unet":
                model = UNet(**self.network_config["unet"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be unet only.")
                
            ddpm = DDPM(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["ddpm"])
                
            self.learner = Learner(self.result_folder,
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
                model = UNet_conditional(**self.network_config["conditional_unet"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be conditional_unet only.")
                
            ddpm = DDPM(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["ddpm"])
                
            self.learner = Conditional_Learner(self.result_folder,
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
                model = UNet_Latent(**self.network_config["unet_latent"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be unet_latent, or en_de_latent only.")
                
            latent_dm = Latent_Diffusion(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["latent"])
                
            self.learner = Learner_Latent(self.result_folder,
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
                model = UNet_conditional_Latent(**self.network_config["conditional_unet_latent"]).to(DEVICE)
                
            else:
                raise ValueError(
                    f"Invalid value for model: {self.learner_config['noise_prediction_model']}, it should be conditional_unet_latent only.")
                
            latent_dm = Latent_Diffusion(model, device=DEVICE, n_steps=self.learner_config["n_steps"], **self.diffusion_config["latent"])
                
            self.learner = Conditional_Learner_Latent(self.result_folder,
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

            
        self.learner.fit(test_epoch_step=self.learner_config["testevery"])

        self.learner.parameter_storage.write("Current config:-\n")
        self.learner.parameter_storage.store(self.config)

        self.learner.parameter_storage.write(
            f"Start Time of gan training and evaluation in this run: {start_time.strftime('%H:%M:%S')}")

        self.learner.parameter_storage.write("\n")

        print("\n\n******* Run is completed*******")

        end_time = datetime.now()

        self.learner.parameter_storage.write(
            f"End Time of gan training and evaluation in this run: {end_time.strftime('%H:%M:%S')}\n")

        self.duration_trial = end_time - start_time
        self.durations=self.duration_trial

        self.learner.parameter_storage.write(
            f"Duration of gan training and evaluation in this run: {str(self.durations)[:-7]}\n")

        return self.learner.best_values["TestLoss"]
    
