
import torch,os
from .msggan_generator import Generator
from .msggan_discriminator import Discriminator
import copy
import time
import numpy as np
import datetime
import timeit
from torch.nn import DataParallel


class MSG_GAN:
    """ Unconditional TeacherGAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            use_eql: whether to use the equalized learning rate
            use_ema: whether to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512,
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=torch.device("cpu")):
        """ constructor for the class """
        

        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)

        self.dis = Discriminator(depth, latent_size, use_eql=True).to(device)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        if self.use_ema:
            from .msggan_layers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = torch.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward()
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :return: current loss
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward()
        gen_optim.step()

        # if self.use_ema is true, apply the moving average here:
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item()



    def train(self, data, gen_optim, dis_optim, loss_fn, normalize_latents=False,
              start=1, num_epochs=12, feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36,
              log_dir=None, sample_dir="./samples",
              save_dir="./models"):
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param loss_fn: Object of GANLoss
        :param normalize_latents: whether to normalize the latent vectors during training
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param save_dir: path to directory for saving the trained models
        :return: None (writes multiple files to disk)
        """

        from torch.nn.functional import avg_pool2d

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()

        assert isinstance(gen_optim, torch.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, torch.optim.Optimizer), \
            "dis_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)
        if normalize_latents:
            fixed_input = (fixed_input
                           / fixed_input.norm(dim=-1, keepdim=True)
                           * (self.latent_size ** 0.5))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch

            print("\nEpoch: %d" % epoch)
            print("#############################################################")
            total_batches = len(iter(data))

            limit = int((data_percentage / 100) * total_batches)

            for (i, batch) in enumerate(data, 1):

                # extract current batch of data for training
                images = batch.to(self.device)
                extracted_batch_size = images.shape[0]

                # create a list of downsampled images from the real images:
                images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                                     for i in range(1, self.depth)]
                images[0].requires_grad = True
                images = list(reversed(images))

                # sample some random latent points
                gan_input = th.randn(
                    extracted_batch_size, self.latent_size).to(self.device)

                # normalize them if asked
                if normalize_latents:
                    gan_input = (gan_input
                                 / gan_input.norm(dim=-1, keepdim=True)
                                 * (self.latent_size ** 0.5))

                # optimize the discriminator:
                dis_loss = self.optimize_discriminator(dis_optim, gan_input,
                                                       images, loss_fn)

                # optimize the generator:
                gen_loss = self.optimize_generator(gen_optim, gan_input,
                                                   images, loss_fn)

                # provide a loss feedback
                if i % (int(limit / feedback_factor) + 1) == 0 or i == 1:     # Avoid div by 0 error on small training sets
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f"
                          % (elapsed, i, dis_loss, gen_loss))

                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(global_step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\n")

                # increment the global_step:
                global_step += 1

                if i > limit:
                    break

            # create a grid of samples and save it
            reses = [str(int(np.power(2, dep))) + "_x_"
                     + str(int(np.power(2, dep)))
                     for dep in range(2, self.depth + 2)]
            gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                          str(epoch) + "_" + ".png")
                             for res in reses]

            # Make sure all the required directories exist
            # otherwise make them
            os.makedirs(sample_dir, exist_ok=True)
            for gen_img_file in gen_img_files:
                os.makedirs(os.path.dirname(gen_img_file), exist_ok=True)

            dis_optim.zero_grad()
            gen_optim.zero_grad()
            with torch.no_grad():
                self.create_grid(
                    self.gen(fixed_input) if not self.use_ema else self.gen_shadow(fixed_input),
                    gen_img_files)

            # calculate the time required for the epoch
            stop_time = timeit.default_timer()
            print("#############################################################")
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                torch.save(self.gen.state_dict(), gen_save_file)
                torch.save(self.dis.state_dict(), dis_save_file)
                torch.save(gen_optim.state_dict(), gen_optim_save_file)
                torch.save(dis_optim.state_dict(), dis_optim_save_file)

                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_"
                                                        + str(epoch) + ".pth")
                    torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()