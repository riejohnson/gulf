import argparse

from utils.utils0 import raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from gulf import is_gulf
from image.imgnet_train import main as imgnet_train

#----------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--model', default='resnet50pre', type=str,
                       choices=['resnet50pre','wrn50-2pre'], help='Model type.')
   parser.add_argument('--ini_type', default='iniBase/2', type=str, choices=['iniBase','iniBase/2'],
                       help='Parameter initialization method.')
   parser.add_argument('--alpha', default=0.5, type=float, help='Alpha.')
   parser.add_argument('--num_stages', type=int, help='Number of stages.')
   parser.add_argument('--dataroot', type=str, required=True, help='Root directory of the ImageNet data.')
   parser.add_argument('--nthread', default=20, type=int, help='Number of workers for training.')
   parser.add_argument('--nthread_test', default=4, type=int, help='Number of workers for testing.')
   parser.add_argument('--save', default='', type=str, help='Pathname for saving models.')
   parser.add_argument('--resume', default='', type=str, help='Pathname for resuming.')
   parser.add_argument('--csv_fn', default='', type=str, help='Pathname for writing test results in the CSV format.')
   parser.add_argument('--seed', default=1, type=int, help='Random seed.')
   parser.add_argument('--ngpu', type=int, help='Number of GPUs.')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')

#----------------------------------------------------------
def check_args_(opt):
   opt.dtype = 'float'
   #---  gulf
   opt.m = -1
   opt.fc_name = 'fc'         # Name of the last fully-connected layer
   opt.initial = ''; opt.do_iniBase = False
   opt.fc_scale = 0.5 if opt.ini_type.endswith('/2') else -1

   #---  optimization
   if opt.num_stages is None:
      opt.num_stages = 3 if opt.model.startswith('resnet') else 1
   opt.max_count = 90         # Length of each stage.  90 epochs.
   opt.do_count_epochs = True # If ture, the unit of max_count and decay_lr_at is epochs.
   opt.batch_size = 256       # Mini-batch size
   opt.lr = 0.1               # Learning rate
   opt.decay_lr_at = [30, 60] # When to decay the learning rate
   opt.lr_decay_ratio = 0.1   # Learning rate decay ratio
   opt.weight_decay = 0.0001  # Weight decay lambda

   #---  data
   opt.dataset = 'ImageNet'
   opt.do_download = False # If true, download data if it does not exist
   opt.do_augment = 1      # Augment data.
   opt.ndev = 10000        # Size of dev. set held out from training data.
   opt.dev_seed = 7

   #---  testing
   opt.do_top5 = True         # Show top-5 error rates.
   opt.test_inc = 100         # Interval of showing progress of testing
   opt.test_interval = 5      # Test after every 5 epochs
   opt.inc = 100              # Interval of showing progress of training

   #---  number of GPUs
   if opt.ngpu is None:
      opt.ngpu = 4 if opt.model.startswith('wrn') else 2

   #---  check values and display ...
   if is_gulf(opt):
      raise_if_nonpositive_any(opt, ['alpha'])
      show_args(opt, ['ini_type','alpha','m'], 'GULF ------')
   raise_if_nonpositive_any(opt, ['max_count','batch_size','lr','lr_decay_ratio','num_stages'])
   show_args(opt, ['max_count','do_count_epochs', 'batch_size','weight_decay','decay_lr_at','lr_decay_ratio','lr','num_stages'],
             'Optimization  ------')
   raise_if_nonpositive_any(opt, ['ndev'])
   show_args(opt, ['dataset','dataroot','do_augment','do_download','ndev','dev_seed'], 'Data -----')
   show_args(opt, ['nthread','nthread_test'], 'Number of workers -----')
   show_args(opt, ['save','csv_fn','test_interval'], 'Others ------')
   show_args(opt, ['model'], 'Model ---')
   raise_if_nonpositive_any(opt, ['ngpu'])
   show_args(opt, ['seed','ngpu','verbose'], 'Miscellaneous ---')

#********************************************************************
def main():
   parser = ArgParser_HelpWithDefaults(description='train_imgnet', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args()
   check_args_(opt)

   imgnet_train(opt)

if __name__ == '__main__':
   main()
