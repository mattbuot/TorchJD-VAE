import argparse
import os
import random as rand
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import VAEDataset
from experiment import VAEXperiment
from torchjd_vae.models import *

#from pytorch_lightning.plugins import DDPPlugin

def main(config_file):
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default=config_file)

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    experiment_name = "GS5_" + config["model_params"]["name"] + '_' + \
                        config["exp_params"]["backward_option"] + '_' + \
                        str(config["model_params"]["latent_dim"]) + '_' + \
                        f"{config['exp_params']['LR']}" + '_' + \
                        f"{config['exp_params']['kld_weight']:e}" + '_' + \
                        f"{config['exp_params']['kld_preferred_weight']}" + '_' + \
                        f"{config['exp_params']['aggregator']}"
    tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                name=experiment_name,)
    # Log hyperparameters to TensorBoard
    tb_logger.log_hyperparams(config)

    # For reproducibility
    rand.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = VAEXperiment(model,
                            config['exp_params'])

    pin_memory = len(config['trainer_params'].get("gpus", [])) != 0
    data = VAEDataset(**config["data_params"], pin_memory=pin_memory)

    data.setup()
    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    #strategy=DDPPlugin(find_unused_parameters=False),
                    **config['trainer_params'])


    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)


if __name__ == "__main__":
    for config_file in [
        # "configs/grid_search2/mnist_vae0.yaml",
        # "configs/grid_search2/mnist_vae1.yaml",
        # "configs/grid_search2/mnist_vae2.yaml",
        # "configs/grid_search2/mnist_vae3.yaml",
        # "configs/grid_search2/mnist_vae4.yaml",
        # "configs/grid_search2/mnist_vae5.yaml",
        # "configs/grid_search2/mnist_vae6.yaml",
        # "configs/grid_search2/mnist_vae7.yaml",
        # "configs/grid_search2/mnist_vae8.yaml",
        # "configs/grid_search2/mnist_vae9.yaml",
        # "configs/grid_search2/mnist_vae10.yaml",
        # "configs/grid_search2/mnist_vae11.yaml",
        # "configs/grid_search2/mnist_vae12.yaml",
        # "configs/grid_search2/mnist_vae13.yaml",
        # "configs/grid_search2/mnist_vae14.yaml",
        #"configs/grid_search2/mnist_vae15.yaml",
        # "configs/grid_search2/mnist_vae16.yaml",
        # "configs/grid_search2/mnist_vae17.yaml",
        # "configs/grid_search2/mnist_vae18.yaml",
        # "configs/grid_search2/mnist_vae19.yaml",
       # "configs/grid_search2/mnist_vae20.yaml",
        # "configs/grid_search2/mnist_vae21.yaml",
        # "configs/grid_search2/mnist_vae22.yaml",
        # "configs/grid_search2/mnist_vae23.yaml",
        # "configs/grid_search2/mnist_vae24.yaml",
        # "configs/grid_search2/mnist_vae25.yaml",
        # "configs/grid_search2/mnist_vae15.yaml",
        #"configs/grid_search2/mnist_vae26.yaml",
        #"configs/grid_search2/mnist_vae27.yaml",
        #"configs/grid_search2/mnist_vae28.yaml",
        "configs/grid_search2/mnist_vae30.yaml",
        "configs/grid_search2/mnist_vae31.yaml",
        "configs/grid_search2/mnist_vae32.yaml",
        "configs/grid_search2/mnist_vae29.yaml",

    ]:
        main(config_file)