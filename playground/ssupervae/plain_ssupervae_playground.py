import os
import sys
import json

from torch import optim
from torch.utils.data import DataLoader

from data.datasets.random_dataset import RandomDataset
from data.datasets.golden_panels import GoldenPanelsDataset

from networks.plain_ssupervae import PlainSSuperVAE
from training.vae_trainer import VAETrainer
from utils.config_utils import read_config, Config
from utils.plot_utils import *
from utils.logging_utils import *
from utils import pytorch_util as ptu

from configs.base_config import *
from functional.losses.elbo import elbo


def save_best_loss_model(model_name, model, best_loss):
    print('[INFO] Current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/ssupervae/weights/' + model_name + ".pth")

def train(data_loader, config, model_name='plain_ssupervae'):
    # loading config
    print("[INFO] Initiate training...")

    # creating model and training details
    net = PlainSSuperVAE(config.backbone, 
                         latent_dim=config.latent_dim, 
                         embed_dim=config.embed_dim,
                         seq_size=config.seq_size,
                         decoder_channels=config.decoder_channels,
                         gen_img_size=config.image_dim).to(ptu.device)
    criterion = elbo

    optimizer = optim.Adam(net.parameters(),
                           lr=config.lr,
                           betas=(config.beta_1, config.beta_2),
                           weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                            last_epoch=-1)
    # init trainer
    trainer = VAETrainer(model=net,
                         model_name=model_name,
                         criterion=criterion,
                         train_loader=data_loader,
                         test_loader=None,
                         epochs=config.train_epochs,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         grad_clip=config.g_clip,
                         best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l),
                         save_dir='playground/ssupervae/',
                         checkpoint_every_epoch=True
                        )
    train_losses, test_losses = trainer.train_epochs()

    print("[INFO] Completed training!")
    
    save_training_plot(train_losses['loss'],
                       test_losses['loss'],
                       "Plain_SSuperVAE Losses",
                       base_dir + 'playground/supervae/' + f'results/ssupervae_plot.png'
                      )
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    config = read_config(Config.PLAIN_SSUPERVAE)
    golden_age_config = read_config(Config.GOLDEN_AGE)
    
    # data = RandomDataset((3, 3, 360, 360), (3, config.image_dim, config.image_dim))
    data = GoldenPanelsDataset(golden_age_config.panel_path,
                               golden_age_config.sequence_path, 
                               golden_age_config.panel_dim,
                               config.image_dim, 
                               augment=False, 
                               mask_val=1,
                               mask_all=False,
                               limit_size=-1)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    
    model = train(data_loader, config, get_dt_string() + "_model")
    torch.save(model, base_dir + 'playground/ssupervae/results/' + "ssuper_vae_model.pth")