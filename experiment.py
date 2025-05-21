import os
import math
import torch
from torch import optim
from torchjd_vae.models import BaseVAE
from torchjd_vae.models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from enum import Enum
from torchjd import mtl_backward, backward
from torchjd.aggregation import UPGrad, Sum, Mean

def print_weights(_, __, weights: torch.Tensor) -> None:
    """Prints the extracted weights."""
    print(f"Weights: {weights}")

def print_similarity_with_gd(_, inputs: tuple[torch.Tensor], aggregation: torch.Tensor) -> None:
    """Prints the cosine similarity between the aggregation and the average gradient."""
    matrix = inputs[0]
    gd_output = matrix.mean(dim=0)
    similarity = cosine_similarity(aggregation, gd_output, dim=0)
    print(f"Cosine similarity: {similarity.item():.4f}")

def print_aggregated_gradients(_, __, aggregation: torch.Tensor) -> None:
    """Prints the aggregated gradients."""
    print(f"Aggregated gradients: {aggregation}")

def print_jacobian(_, inputs: tuple[torch.Tensor], __) -> None:
    """Prints the Jacobian matrix."""
    jacobian = inputs[0]
    print(f"Jacobian: {jacobian}")

global_step = 0

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.automatic_optimization = False

        self.aggregator = UPGrad(pref_vector=torch.Tensor([1, params['kld_preferred_weight']]).to(self.device),)
        #self.aggregator.weighting.register_forward_hook(print_weights)
        self.aggregator.register_forward_hook(self.log_cosine_similarity)
        #self.aggregator.register_forward_hook(print_jacobian)
        #self.aggregator.register_forward_hook(print_aggregated_gradients)

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):

        global global_step

        real_img, _ = batch
        self.curr_device = real_img.device


        mu, log_var = self.model.encode(real_img)
        z = self.model.reparameterize(mu, log_var)
        results = [self.model.decode(z), real_img, mu, log_var]

        loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              #optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        
        train_loss = loss['loss']
        reconstruction_loss = loss['Reconstruction_Loss']
        kld_loss = loss['KLD'] * self.params['kld_weight']

        opt = self.optimizers()
        opt.zero_grad()

        class BackwardOptions(str, Enum):
            """Enum for backward options."""
            TORCH = "torch"
            TORCH_JD_BACKWARD = "torch_jd_backward"
            TORCH_JD_MTL_BACKWARD = "torch_jd_mtl_backward"


        backward_option = self.params.get('backward_option', BackwardOptions.TORCH)

        if backward_option == BackwardOptions.TORCH:
            train_loss.backward()
        elif backward_option == BackwardOptions.TORCH_JD_BACKWARD:
            backward(tensors=[reconstruction_loss, kld_loss],
                     aggregator=self.aggregator,
                     parallel_chunk_size=1,)
        elif backward_option == BackwardOptions.TORCH_JD_MTL_BACKWARD:
            mtl_backward(losses=[reconstruction_loss, kld_loss],
                            features=[mu, log_var],
                            aggregator=self.aggregator,
                            parallel_chunk_size=1,)
        else:
            raise ValueError("Invalid backward option")
        
        opt.step()

        self.log_dict({key: val.item() for key, val in loss.items()}, sync_dist=True)

        # if batch_idx % 1000 == 0:
        #     self.logger.experiment.add_histogram('Distribution of mu', mu.detach().cpu().numpy(), global_step=global_step, bins="rice")
        #     self.logger.experiment.add_histogram('Distribution of log_var', log_var.detach().cpu().numpy(), global_step=global_step, bins="rice")
        #     self.logger.experiment.add_histogram('Distribution of z', z.detach().cpu().numpy(), global_step=global_step, bins="rice")
        #     global_step += 1

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            #optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def log_cosine_similarity(self, _, inputs: tuple[torch.Tensor], aggregation: torch.Tensor) -> None:
        """Logs the cosine similarity between the aggregation and the average gradient."""
        matrix = inputs[0]
        gd_output = matrix.mean(dim=0)
        similarity = cosine_similarity(aggregation, gd_output, dim=0)
        self.log_dict({"Cosine Similarity": similarity.item()}, sync_dist=True)


    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
