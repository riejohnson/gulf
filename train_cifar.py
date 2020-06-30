import argparse

from utils.utils0 import raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from gulf import is_gulf, interpret_ini_type_
from image.resnet_train import main as resnet_train

#----------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--k', default=1, choices=[1,10], type=float, help='Network width.')
   parser.add_argument('--alpha', default=0.3, type=float, help='Alpha.')
   parser.add_argument('--m', default=-1, type=int, help='m for GULF1. -1: Do GULF2.')
   parser.add_argument('--ini_type', default='iniRand', type=str, choices=['iniRand','iniBase','iniBase/2','file','file/2'],
                       help='Parameter initialization method.')
   parser.add_argument('--initial', default='', type=str, help='Pathname of the initial model.  Use this when ini_type==file or file/2.')
   parser.add_argument('--num_stages', type=int, help='Number of stages.')
   parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10','CIFAR100'], type=str, help='Dataset name.')
   parser.add_argument('--dataroot', default='.', type=str, help='Root directory of the data.')
   parser.add_argument('--nthread', default=4, type=int, help='Number of workers for training.')
   parser.add_argument('--nthread_test', default=0, type=int, help='Number of workers for testing.')
   parser.add_argument('--resume', default='', type=str, help='Model pathname for resuming training from.')
   parser.add_argument('--save', default='', type=str, help='Pathname for saving models.')
   parser.add_argument('--csv_fn', default='', type=str, help='Pathname for writing test results in the CSV format.')
   parser.add_argument('--seed', default=1, type=int, help='Random seed.')
   parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs.')
   parser.add_argument('--do_collect_info', action='store_true', help='Compute training loss and parameter norm.')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')

#----------------------------------------------------------
def check_args_(opt):
   opt.dtype = 'float'
   interpret_ini_type_(opt)

   #---  Model
   opt.depth = 28   # Network depth.
   opt.dropout = -1 # No dropout

   #---  Optimization and GULF
   opt.fc_name = 'fc'
   if opt.num_stages is None:
      opt.num_stages = 25 if opt.k == 1 else 1
   opt.weight_decay = 0.0005 if opt.dataset == 'CIFAR100' and opt.k == 10 else 0.0001
   opt.max_count = 280000        # Length of each stage.  280K steps.
   opt.do_count_epochs = False
   opt.batch_size = 128          # Mini-batch size
   opt.lr = 0.1                  # Learning rate
   opt.decay_lr_at = [200000, 240000] # Decay the learning rate after 200K and 240K steps
   opt.lr_decay_ratio = 0.1      # To decay the learning rate, multiply 0.1.
   opt.max_grad_norm = -1

   #---  Data
   opt.do_download = True # Download data if it does not exist
   opt.do_augment = 1     # Augment data.
   opt.ndev = 1000        # Size of the dev. set held out from training data
   opt.dev_seed = 7       # Random seed for generating the dev. set

   #---  Testing
   opt.do_reduce_testing = True  # Evaluate on the test set only at the end of each stage.
   opt.test_interval = 40000     # Test interval
   opt.inc = 100                 # Interval of showing progress of training.

   #---  Check values and display ...
   raise_if_nonpositive_any(opt, ['k','depth','ngpu'])
   show_args(opt, ['depth','k','dropout'], 'Model ---')
   raise_if_nonpositive_any(opt, ['num_stages','max_count','batch_size','lr','lr_decay_ratio'])
   show_args(opt, ['max_count','do_count_epochs', 'batch_size','decay_lr_at','lr_decay_ratio','lr',
                   'weight_decay','num_stages','max_grad_norm'], 'Optimization ------')
   if is_gulf(opt):
      show_args(opt, ['ini_type','alpha','m','initial'], 'GULF training ------')
   raise_if_nonpositive_any(opt, ['ndev'])
   show_args(opt, ['dataset','dataroot','do_augment','do_download','ndev'], 'Data -----')
   show_args(opt, ['nthread','nthread_test'], 'Number of workers -----')
   show_args(opt, ['save','resume','seed','ngpu','verbose','csv_fn','test_interval','do_collect_info'], 'Others ------')

#********************************************************************
def main():
   parser = ArgParser_HelpWithDefaults(description='train_cifar', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args()
   check_args_(opt)

   resnet_train(opt)

if __name__ == '__main__':
   main()
