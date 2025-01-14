from torch import optim
from torch.utils.data import DataLoader

from data.datasets.ffhq_dataset import FFHQDataset
from data.datasources.ffhq_datasource import FFHQDatasource
from data.datasources.golden_age_face_datasource import GoldenAgeFaceDatasource
from functional.losses.bi_discriminator_loss import BidirectionalDiscriminatorLoss, BidirectionalDiscriminatorLossType
from networks.bigan import BiGAN
from networks.w_bigan import WBiGAN
from training.bigan_trainer import BiGANTrainer
from utils.config_utils import read_config, Config
from utils.logging_utils import *
from utils.plot_utils import *

from data.datasets import facedataset
from data.datasources import facedatasource
from data.datasources.datasource_mode import DataSourceMode
from networks.siamese_network import SiameseNetwork
from functional.losses.contrastive_loss import ContrastiveLoss
from functional.metrics.dissimilarity import *
from training.face_recognition_trainer import train_epochs
from configs.base_config import *


def visualize_data():
    config = read_config(Config.BiGAN)
    dataset = FFHQDataset(
        datasource=FFHQDatasource(config, mode=DataSourceMode.TRAIN))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    dataiter = iter(dataloader)
    example_batch = next(dataiter)
    imshow(torchvision.utils.make_grid(example_batch, nrow=4))


def visualize_golden_age_face_data():
    train_config = read_config(Config.BiGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE_FACE)
    train_dataset = FFHQDataset(
        datasource=GoldenAgeFaceDatasource(golden_age_config, mode=DataSourceMode.TRAIN))
    dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    dataiter = iter(dataloader)
    example_batch = next(dataiter)
    imshow(torchvision.utils.make_grid(example_batch, nrow=4))


def save_best_loss_model(model_name, model, best_loss):
    # print('current best loss: ' + str(best_loss))
    logging.info('current best loss: ' + str(best_loss))
    torch.save(model, base_dir + 'playground/bigan/results/' + model_name + ".pth")


def train_bigan(model_name='test_model'):
    logging.info("initiate training")
    config = read_config(Config.BiGAN)
    train_dataset = FFHQDataset(
        datasource=FFHQDatasource(config, mode=DataSourceMode.TRAIN))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    net = BiGAN(image_dim=config.image_dim, latent_dim=config.latent_dim_z).to(ptu.device)
    criterion = BidirectionalDiscriminatorLoss(loss_type=BidirectionalDiscriminatorLossType.VANILLA_LOG_MEAN)

    d_optimizer = torch.optim.Adam(net.discriminator.parameters(),
                                   lr=config.discriminator_lr,
                                   betas=(config.discriminator_beta_1, config.discriminator_beta_2),
                                   weight_decay=config.discriminator_weight_decay)

    g_optimizer = torch.optim.Adam(list(net.encoder.parameters()) + list(net.generator.parameters()),
                                   lr=config.generator_lr,
                                   betas=(config.generator_beta_1, config.generator_beta_2),
                                   weight_decay=config.generator_weight_decay)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    trainer = BiGANTrainer(model=net,
                           criterion=criterion,
                           train_loader=train_dataloader,
                           test_loader=None,
                           epochs=config.train_epochs,
                           optimizer_generator=g_optimizer,
                           optimizer_discriminator=d_optimizer,
                           scheduler_gen=g_scheduler,
                           scheduler_disc=d_scheduler,
                           best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))
    losses = trainer.train_bigan()

    logging.info("completed training")
    save_training_plot(losses['discriminator_loss'],
                       losses['generator_loss'],
                       "BiGAN Losses",
                       base_dir + 'playground/bigan/' + f'results/bigan_plot.png')
    return net


def train_w_bigan(model_name='test_model'):
    logging.info("initiate training")
    config = read_config(Config.BiGAN)
    train_dataset = FFHQDataset(
        datasource=FFHQDatasource(config, mode=DataSourceMode.TRAIN))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    net = WBiGAN(image_dim=config.image_dim, latent_dim=config.latent_dim_z).to(ptu.device)
    criterion = BidirectionalDiscriminatorLoss(loss_type=BidirectionalDiscriminatorLossType.WASSERSTEIN)

    d_optimizer = optim.RMSprop(net.discriminator.parameters(), lr=config.discriminator_lr)  # \
    # torch.optim.Adam(net.discriminator.parameters(),
    #                          lr=config.discriminator_lr,
    #                     betas=(config.discriminator_beta_1, config.discriminator_beta_2),
    #                           weight_decay=config.discriminator_weight_decay)

    g_optimizer = optim.RMSprop(list(net.encoder.parameters()) + list(net.generator.parameters()),
                                lr=config.generator_lr)  # \
    # torch.optim.Adam(list(net.encoder.parameters()) + list(net.generator.parameters()),
    #                      lr=config.generator_lr,
    #                     betas=(config.generator_beta_1, config.generator_beta_2),
    #                      weight_decay=config.generator_weight_decay)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
                                                    lambda epoch: (config.train_epochs - epoch) / config.train_epochs,
                                                    last_epoch=-1)
    trainer = BiGANTrainer(model=net,
                           criterion=criterion,
                           train_loader=train_dataloader,
                           test_loader=None,
                           epochs=config.train_epochs,
                           optimizer_generator=g_optimizer,
                           optimizer_discriminator=d_optimizer,
                           scheduler_gen=g_scheduler,
                           scheduler_disc=d_scheduler,
                           disc_weight_clipping=(-0.01, 0.01),
                           generator_update_round=5,
                           best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))
    losses = trainer.train_bigan()

    logging.info("completed training")
    save_training_plot(losses['discriminator_loss'],
                       losses['generator_loss'],
                       "BiGAN Losses",
                       base_dir + 'playground/bigan/' + f'results/bigan_plot.png')
    return net


def train_golden_age_face_bigan(model_name='test_model'):
    logging.info("initiate training")
    train_config = read_config(Config.BiGAN)
    golden_age_config = read_config(Config.GOLDEN_AGE_FACE)
    train_dataset = FFHQDataset(
        datasource=GoldenAgeFaceDatasource(golden_age_config, mode=DataSourceMode.TRAIN))
    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    net = BiGAN(image_dim=golden_age_config.image_dim).to(ptu.device)
    criterion = BidirectionalDiscriminatorLoss(loss_type=BidirectionalDiscriminatorLossType.VANILLA_LOG_MEAN)

    d_optimizer = torch.optim.Adam(net.discriminator.parameters(),
                                   lr=train_config.discriminator_lr,
                                   betas=(train_config.discriminator_beta_1, train_config.discriminator_beta_2),
                                   weight_decay=train_config.discriminator_weight_decay)

    g_optimizer = torch.optim.Adam(list(net.encoder.parameters()) + list(net.generator.parameters()),
                                   lr=train_config.generator_lr,
                                   betas=(train_config.generator_beta_1, train_config.generator_beta_2),
                                   weight_decay=train_config.generator_weight_decay)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,
                                                    lambda epoch: (
                                                                          train_config.train_epochs - epoch) / train_config.train_epochs,
                                                    last_epoch=-1)
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,
                                                    lambda epoch: (
                                                                          train_config.train_epochs - epoch) / train_config.train_epochs,
                                                    last_epoch=-1)
    trainer = BiGANTrainer(model=net,
                           criterion=criterion,
                           train_loader=train_dataloader,
                           test_loader=None,
                           epochs=train_config.train_epochs,
                           optimizer_generator=g_optimizer,
                           optimizer_discriminator=d_optimizer,
                           scheduler_gen=g_scheduler,
                           scheduler_disc=d_scheduler,
                           best_loss_action=lambda m, l: save_best_loss_model(model_name, m, l))
    losses = trainer.train_bigan()

    logging.info("completed training")
    save_training_plot(losses['discriminator_loss'],
                       losses['generator_loss'],
                       "Golden Age Face BiGAN Losses",
                       base_dir + 'playground/bigan/' + f'results/bigan_plot.png')
    return net


if __name__ == '__main__':
    ptu.set_gpu_mode(True)
    # visualize_data()
    # visualize_golden_age_face_data()
    # model = train_bigan(get_dt_string() + "_model")
    model = train_w_bigan(get_dt_string() + "_model")
    # model = train_golden_age_face_bigan(get_dt_string() + "_model")
    # torch.save(model, base_dir + 'playground/bigan/results/' + "test_model.pth")
    # model = torch.load("test_model.pth")
    # TODO: add visualizations of reconstruction and sampling from model
