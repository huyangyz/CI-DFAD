import torch
from load_data import data_loader
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import logging

class Tester(object):
    def __init__(self, opt):

        self.opt = opt

        self.loader = data_loader(opt)
        self.generator = data.DataLoader(self.loader, batch_size=opt['batch_size'], shuffle=True, drop_last=False)

        # model
        self.G = torch.load(self.opt['save_path'] + 'G_' + str(self.opt['epoch']) + '.pth')
        self.D = torch.load(self.opt['save_path'] + 'D_' + str(self.opt['epoch']) + '.pth')

        # loss function
        self.G_loss = nn.MSELoss()
        self.D_loss = nn.BCELoss()

        if opt['cuda']:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.G_loss = self.G_loss.cuda()
            self.D_loss = self.D_loss.cuda()

    def test(self):

        self.G.eval()
        self.D.eval()
        result = torch.zeros((self.loader.time_num, self.loader.node_num, 4))

        mc_samples = 100
        eps = 1e-6

        from torch.nn.modules.dropout import _DropoutNd
        def _set_dropout_mode(model, mode='train'):
            for m in model.modules():
                if isinstance(m, _DropoutNd):
                    if mode == 'train':
                        m.train()
                    else:
                        m.eval()
                else:
                    if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                        if mode == 'train':
                            m.train()
                        else:
                            m.eval()

        for step, (
                (recent_data, trend_data, day_data, time_feature), sub_graph, real_data, index_t, index_r) in enumerate(
            self.generator):

            if self.opt['cuda']:
                recent_data = recent_data.cuda()
                trend_data = trend_data.cuda()
                day_data = day_data.cuda()
                real_data = real_data.cuda()
                sub_graph = sub_graph.cuda()
                time_feature = time_feature.cuda()

            real_sequence = torch.cat([recent_data, real_data.unsqueeze(1)], dim=1)

            single_fake = self.G(recent_data, trend_data, day_data, sub_graph, time_feature)
            fake_sequence_single = torch.cat([recent_data, single_fake.unsqueeze(1)], dim=1)


            with torch.no_grad():

                _set_dropout_mode(self.G, 'train')
                samples = []
                for _ in range(mc_samples):
                    s = self.G(recent_data, trend_data, day_data, sub_graph, time_feature)  # (batch, node, feat)
                    samples.append(s.unsqueeze(0))
                samples = torch.cat(samples, dim=0)

                _set_dropout_mode(self.G, 'eval')


            mu = samples.mean(dim=0)
            std = samples.std(dim=0)

            residual_abs = torch.abs(real_data - mu)
            anomaly_per_feat = residual_abs / (std + eps)

            mse_mu_per_sample = ((mu - real_data) ** 2).mean(dim=2).mean(dim=1)
            uncertainty_score_batch = anomaly_per_feat.mean(dim=2).mean(dim=1)

            fake_sequence_mu = torch.cat([recent_data, mu.unsqueeze(1)], dim=1)

            real_score_D = self.D(real_sequence, sub_graph)
            fake_score_D = self.D(fake_sequence_mu, sub_graph)

            batch_size = recent_data.shape[0]
            for b in range(batch_size):
                t_idx = index_t[b].item()
                r_idx = index_r[b].item()

                val0 = mse_mu_per_sample[b]
                result[t_idx, r_idx, 0] = float(val0.item() if torch.is_tensor(val0) else val0)

                if torch.is_tensor(real_score_D):
                    r_val = real_score_D[b]
                    result[t_idx, r_idx, 1] = float(r_val.mean().item() if r_val.dim() > 0 else r_val.item())
                else:
                    result[t_idx, r_idx, 1] = float(real_score_D)

                if torch.is_tensor(fake_score_D):
                    f_val = fake_score_D[b]
                    result[t_idx, r_idx, 2] = float(f_val.mean().item() if f_val.dim() > 0 else f_val.item())
                else:
                    result[t_idx, r_idx, 2] = float(fake_score_D)

                val3 = uncertainty_score_batch[b]
                result[t_idx, r_idx, 3] = float(val3.item() if torch.is_tensor(val3) else val3)

            if step % 100 == 0:
                logging.info("step:%d [G mse(mu): %f]" % (step, mse_mu_per_sample.mean().item()))

        np.save(self.opt['result_path'] + 'result' + '.npy', result.cpu().numpy())



