import torch
from torch.distributions.normal import Normal

# Models

from networks.panel_encoder.plain_sequential_encoder import PlainSequentialEncoder
from networks.panel_encoder.lstm_sequential_encoder import LSTMSequentialEncoder
from networks.dcgan import DCGAN
# Helpers
from utils import pytorch_util as ptu

import torch.nn as nn
from torchvision.utils import save_image


class SSuperMSGGAN(nn.Module):

    def __init__(self,
                 # common parameters
                 backbone,
                 latent_dim=256,
                 embed_dim=256,
                 use_lstm=False,
                 # plain encoder parameters
                 seq_size=3,
                 gen_img_size=64,
                 # lstm encoder parameters
                 lstm_hidden=256,
                 lstm_dropout=0,
                 fc_hidden_dims=[],
                 fc_dropout=0,
                 num_lstm_layers=1,
                 masked_first=True,
                 depth=7,
                 use_eql=True,
                 use_ema=True,
                 ema_decay=0.999,
                 device=torch.device("cpu")

                 ):
        super(SSuperMSGGAN, self).__init__()

        self.latent_dim = latent_dim

        if not use_lstm:
            self.encoder = PlainSequentialEncoder(backbone,
                                                  latent_dim=latent_dim,
                                                  embed_dim=embed_dim,
                                                  seq_size=seq_size)
        else:
            self.encoder = LSTMSequentialEncoder(backbone,
                                                 latent_dim=latent_dim,
                                                 embed_dim=embed_dim,
                                                 lstm_hidden=lstm_hidden,
                                                 lstm_dropout=lstm_dropout,
                                                 fc_hidden_dims=fc_hidden_dims,
                                                 fc_dropout=fc_dropout,
                                                 num_lstm_layers=num_lstm_layers,
                                                 masked_first=masked_first)

        from .MSGGAN.msggan import MSG_GAN

        self.depth = depth

        self.gan = MSG_GAN(depth, latent_dim, use_eql=use_eql, use_ema = use_ema, ema_decay= ema_decay, device=ptu.device)

        self.latent_dist = Normal(
            ptu.FloatTensor([0.0], torch_device=ptu.device),
            ptu.FloatTensor([1.0], torch_device=ptu.device)
        )

    def forward(self, x):
        mu, lg_std = self.encode(x)
        z = torch.distributions.Normal(mu, lg_std.exp()).rsample()
        
        #print("Z shape",z.shape, "X shape ",x.shape)
        #z = torch.unsqueeze(z, (2))
        #z = torch.unsqueeze(z, (3))
        outputs,_ = self.gan.gen(z)
        
        return z, None, mu, outputs, lg_std
        

    def encode(self, x):
        return self.encoder(x)

    def generate(self, x):
        mu, _ = self.encode(x)
        return mu

    def decode(self, z):
        outputs,_ = self.gan.gen(z)
        #print("OUtputs", len(outputs))
        #print(gen_out)
        return outputs[-1]

    def sample(self, size: int):
        z = self.latent_dist.rsample((size, self.latent_dim)).squeeze(-1)
        #print("Z shape ")
        #z = torch.unsqueeze(z, (2))
        #print("Forward z shape ",z.shape)
        #z = torch.unsqueeze(z, (3))
        # print("Sample z size : ",z.shape)
        return self.decode(z)

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu)

    @torch.no_grad()
    def save_samples(self, n, filename):
        samples = self.sample(size=n)
        save_image(samples, filename, nrow=10, normalize=True)
