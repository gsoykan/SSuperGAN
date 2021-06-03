from torch.nn.functional import avg_pool2d
from networks.ssuper_msggan import SSuperMSGGAN
import functional.losses as lses
import os
import sys
import json
os.path.dirname(sys.executable)
sys.path.append("/kuacc/users/ckoksal20/COMP547Project/SSuperGAN/")

import torch
from torch import optim
from torch.utils.data import DataLoader
from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_faces import GoldenFacesDataset
from data.datasets.golden_panels import GoldenPanelsDataset

from training.dcgan_trainer import DCGANTrainer
from training.ssuper_dcgan_trainer import SSuperDCGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *
from utils import pytorch_util as ptu
from configs.base_config import *

from functional.losses import RelativisticAverageHingeGAN
import torch.nn as nn

from collections import OrderedDict
from functional.losses.elbo import elbo
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim

from networks.base.base_vae import BaseVAE
from networks.generic_vae import GenericVAE
from networks.ssuper_dcgan import SSuperDCGAN
from training.base_trainer import BaseTrainer
from utils.structs.metric_recorder import *
from utils.logging_utils import *
from utils import pytorch_util as ptu


class SSuperMSGGANTrainer(BaseTrainer):
    def __init__(self,
                 model: SSuperMSGGAN,
                 model_name: str,
                 criterion,
                 train_loader,
                 test_loader,
                 epochs: int,
                 optimizer_encoder,
                 optimizer_generator,
                 optimized_discriminator,
                 scheduler=None,
                 quiet: bool = False,
                 grad_clip=None,
                 best_loss_action=None,
                 save_dir=base_dir + 'playground/SSuperDCGAN/',
                 checkpoint_every_epoch=False):
        super().__init__(model,
                         model_name,
                         criterion,
                         epochs,
                         save_dir,
                         {
                            "optimizer_encoder": optimizer_encoder,
                            "optimizer_generator": optimizer_generator,
                            "optimizer_discriminator":optimized_discriminator},
                         
                         {"scheduler": scheduler},
                         quiet,
                         grad_clip,
                         best_loss_action,
                         checkpoint_every_epoch)
        self.train_loader = train_loader
        self.test_loader = test_loader
        


        

    def train_epochs(self, starting_epoch=None, losses={}):
        metric_recorder = MetricRecorder(experiment_name=self.model_name,
                                         save_dir=self.save_dir + '/results/')
        # TODO: becareful about best loss here this might override the actual best loss
        #  in case of continuation of training
        best_loss = 99999999

        train_losses = losses.get("train_losses", OrderedDict())
        test_losses = losses.get("test_losses", OrderedDict())
        #torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epochs):
            if starting_epoch is not None and starting_epoch >= epoch:
                continue
            logging.info("epoch start: " + str(epoch))
            train_loss = self.train_ssuper_msggan(epoch)

            if self.test_loader is not None:
                test_loss = self.eval_loss(self.test_loader)
            else:
                test_loss = {"loss": 0, "kl_loss": 0, "reconstruction_loss": 0, "disc_loss":0, "gen_loss":0}

            for k in train_loss.keys():
                if k not in train_losses:
                    train_losses[k] = []
                    test_losses[k] = []
                train_losses[k].extend(train_loss[k])
                test_losses[k].append(test_loss[k])
                if k == "loss":
                    current_test_loss = test_loss[k]
                    if current_test_loss < best_loss:
                        best_loss = current_test_loss
                        if self.best_loss_action != None:
                            self.best_loss_action(self.model, best_loss)

            if self.checkpoint_every_epoch:
                self.save_checkpoint(current_loss=
                {
                    "train_losses": train_losses,
                    "test_losses": test_losses
                },
                    current_epoch=epoch)
            
            metric_recorder.update_metrics(train_losses, test_losses)
            metric_recorder.save_recorder()
        return train_losses, test_losses

   

    def train_ssuper_msggan(self, epoch):
        self.model.train()
        if not self.quiet:
            pbar = tqdm(total=len(self.train_loader.dataset))
        losses = OrderedDict()
        for batch in self.train_loader:
            
            if type(batch) == list and len(batch) == 2:
                x, y = batch[0].to(ptu.device), batch[1].to(ptu.device)
                # Shape is 
                #torch.Size([BATCH_SIZE, 3, 3, 300, 300])
                #torch.Size([BATCH_SIZE, 3, 64, 64])
            else:
                x, y = batch.to(ptu.device), None
            
            
            
            self.optimizers["optimizer_encoder"].zero_grad()
            self.optimizers["optimizer_discriminator"].zero_grad()
            self.optimizers["optimizer_generator"].zero_grad()
            
            z, _, mu_z, mu_x, logstd_z = self.model(x)
            
            target = x if y is None else y

            images = [target] + [avg_pool2d(target, int(np.power(2, i)))
                                     for i in range(1, self.depth)]

            images[0].requires_grad = True
            images = list(reversed(images))


            out = elbo(z, target, mu_z, mu_x, logstd_z, l1_recon=False)

            reconstruction_loss = out["reconstruction_loss"]/10
            kl_loss = out["kl_loss"]
            total_loss = out["loss"]

            
            #Optimize Discriminator
            # generate a batch of samples
            
            fake_samples = list(map(lambda x: x.detach(), mu_x))

            disc_loss = self.criterion.dis_loss(images, fake_samples)

            # optimize discriminator
            self.optimizers["optimizer_discriminator"].zero_grad()
            disc_loss.backward()
            self.optimizers["optimizer_discriminator"].step()


            #

            # optimize the generator:


            gen_loss = self.criterion.gen_loss(images, fake_samples)

            gen_loss = gen_loss + reconstruction_loss
            # optimize discriminator
            self.optimizers["optimizer_generator"].zero_grad()
            gen_loss.backward()
            self.optimizers["optimizer_generator"].step()



            # UPDATE ENCODER
            total_loss.backward(retain_graph=True)

            
    
        
            #self.optimizers["optimizer_encoder"].step()
            desc = f'Epoch {epoch}'
            out["disc_loss"] =  disc_loss
            out["gen_loss"] = gen_loss
            for k, v in out.items():
                if k not in losses:
                    losses[k] = []
                if "gen_loss" not in losses  or "disc_loss" not in losses:
                    losses["disc_loss"] = []
                    losses["gen_loss"] = []
                    
                    
                losses[k].append(v.item())
                
                avg = np.mean(losses[k][-50:])
                desc += f', {k} {avg:.4f}'

            if not self.quiet:
                pbar.set_description(desc)
                pbar.update(x.shape[0])

        #self.scheduler.step()
        self.model.save_samples(100, self.save_dir + '/results/' + f'epoch{epoch}_samples.png')
        if not self.quiet:
            pbar.close()
        return losses






