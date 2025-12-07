import torch
from load_data import data_loader
import torch.utils.data as data
import torch.nn as nn
from CI_DFAD_model import Generator, Discriminator
import logging
import time

class Trainer(object):
    def __init__(self, opt):

        self.opt = opt

        self.generator = data.DataLoader(data_loader(opt), batch_size=opt['batch_size'], shuffle=True)

        # model
        self.G = Generator(opt)
        self.D = Discriminator(opt)

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        if opt['cuda']:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.G_loss = self.G_loss.cuda()
            self.D_loss = self.D_loss.cuda()

        # Optimizer
        self.G_optim = torch.optim.RMSprop(self.G.parameters(), lr=opt['lr'])
        self.D_optim = torch.optim.RMSprop(self.D.parameters(), lr=opt['lr'])

    def train(self):
        self.G.train()
        self.D.train()
        for e in range(1, self.opt['epoch']+1):
            start_time = time.time()
            for step, ((recent_data, trend_data, day_data, time_feature), sub_graph, real_data, _, _) in enumerate(self.generator):
                valid = torch.zeros((real_data.shape[0], 1), dtype=torch.float)
                fake = torch.ones((real_data.shape[0], 1), dtype=torch.float)

                if self.opt['cuda']:
                    recent_data, trend_data, day_data, real_data, sub_graph, time_feature, valid, fake = \
                        recent_data.cuda(), trend_data.cuda(), day_data.cuda(), real_data.cuda(), sub_graph.cuda(), time_feature.cuda(), valid.cuda(), fake.cuda()

                    # Train Discriminator
                    self.D_optim.zero_grad()
                    real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)
                    fake_data = self.G(recent_data, trend_data, day_data, sub_graph, time_feature)
                    fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)

                    real_loss = -self.D(real_sequence, sub_graph, trend_data, day_data).mean()
                    fake_loss = self.D(fake_sequence, sub_graph, trend_data, day_data).mean()
                    D_total = real_loss + fake_loss

                    D_total.backward(retain_graph=True)
                    self.D_optim.step()

                    # Train Generator
                    for p in self.D.parameters():
                        p.data.clamp_(-0.05, 0.05)  # weight clipping

                    self.G_optim.zero_grad()
                    fake_data = self.G(recent_data, trend_data, day_data, sub_graph, time_feature)
                    mse_loss = self.G_loss(fake_data, real_data)
                    fake_sequence = torch.cat([recent_data, fake_data.unsqueeze(1)], dim=1)
                    fake_loss = -self.D(fake_sequence, sub_graph, trend_data, day_data).mean()

                    G_total = self.opt['lambda_G'] * mse_loss + fake_loss+ 0.1*self.G.uncertainty_aware_modeling.fc.kl_loss() \
                              + 0.1*self.G.long_term_nonlinear_encoder.nonlinear_encoder_layer.regularization_loss()+0.1*self.G.two_layer_nonlinear_encoder[0].regularization_loss()
                    G_total.backward()
                    self.G_optim.step()

                    if step % 100 == 0:
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        logging.info("epoch:%d step:%d [D loss: %f] [G mse: %f fake_loss %f]" % (
                            e, step, D_total.cpu(), mse_loss, fake_loss))
                        print("epoch:%d step:%d [D loss: %f] [G mse: %f fake_loss %f time: %f]" % (
                            e, step, D_total.cpu(), mse_loss, fake_loss, elapsed_time))

                    torch.cuda.empty_cache()

                    # Save model
                torch.save(self.G, self.opt['save_path'] + 'G_' + str(e) + '.pth')
                torch.save(self.D, self.opt['save_path'] + 'D_' + str(e) + '.pth')