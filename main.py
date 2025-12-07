import argparse
import torch
import numpy as np
import random
from trainer import Trainer
from tester import Tester
import os


torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chicago', help='chicago, nyc or pems')
parser.add_argument('--root_path', type=str, default='./', help='root path: dataset, checkpoint')

parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension.')
# edit defout=6
parser.add_argument('--epoch', type=int, default=6, help='Number of training epochs per iteration.')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lambda_G', type=int, default=500, help='lambda_G for generator loss function')
parser.add_argument('--Diters', type=int, default=5, help='number of epochs to train for')
parser.add_argument('--num_adj', type=int, default=9, help='number of nodes in sub graph')
parser.add_argument('--num_layer', type=int, default=2, help='number of layers in LSTM and DCRNN')
parser.add_argument('--trend_time', type=int, default=7 * 24, help='the length of trend segment is 7 days')
parser.add_argument('--day_time', type=int, default=1 * 24, help='the length of trend segment is 1 days')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda_id', type=str, default='0')
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--cpu', type=bool, default=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# parameter
opt = vars(args)
# 2019-01-01 - 2019-08-31
if opt['dataset'] == 'chicago':
    opt['timestamp'] = 2  # 30min: 2
    opt['train_time'] = 212  # days for training
    opt['recent_time'] = 2  # chicago: 2hour
    opt['num_feature'] = 2 * 2  # length of input feature
    opt['time_feature'] = 31

# 2014-01-15 -- 2014-12-31
elif opt['dataset'] == 'nyc':
    opt['timestamp'] = 2  # 30min: 2
    opt['train_time'] = 289  # days for training
    opt['recent_time'] = 2  # nyc: 2hour
    opt['num_feature'] = 2 * 2  # length of input feature
    opt['time_feature'] = 39


elif opt['dataset'] == 'pems':
    opt['timestamp'] = 12   # 5min: 12
    opt['train_time'] = 105  # days for training
    opt['recent_time'] = 1  # pems: 1 hour
    opt['num_feature'] = 6 * 2 # length of input feature
    opt['time_feature'] = 31

opt['save_path'] = opt['root_path'] + opt['dataset'] + '/checkpoint/'
opt['data_path'] = opt['root_path'] + opt['dataset'] + '/data/'
opt['result_path'] = opt['root_path'] + opt['dataset'] + '/result/'
opt['train_time'] = opt['train_time'] * opt['timestamp'] * 24

if __name__ == "__main__":

    opt['isTrain'] = True
    train_model = Trainer(opt)
    train_model.train()

    opt['isTrain'] = False
    print('test...')
    test_model = Tester(opt)
    test_model.test()


