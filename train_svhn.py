import argparse

from utils.utils0 import raise_if_nonpositive, show_args, raise_if_nonpositive_any, ArgParser_HelpWithDefaults
from gulf import is_gulf, interpret_ini_type_
from image.resnet_train import main as resnet_train

#----------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--dropout', default=0, choices=[0,0.4], type=float, help='Dropout rate.')
   parser.add_argument('--alpha', default=0.3, type=float, help='Alpha.')
   parser.add_argument('--ini_type', default='iniRand', type=str, choices=['iniRand','iniBase','file'], help='Parameter initialization method.')
   parser.add_argument('--initial', default='', type=str, help='Pathname of the initial model.  Use this when ini_type==file.')
   parser.add_argument('--num_stages', type=int, help='Number of stages.')
   parser.add_argument('--dataroot', default='.', type=str, help='Root directory of the data.')
   parser.add_argument('--nthread', default=4, type=int, help='Number of workers for training.')
   parser.add_argument('--nthread_test', default=0, type=int, help='Number of workers for testing.')
   parser.add_argument('--save', default='', type=str, help='Pathname for saving models.')
   parser.add_argument('--csv_fn', default='', type=str, help='Pathname for writing test results in the CSV format.')
   parser.add_argument('--seed', default=1, type=int, help='Random seed.')
   parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs.')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')

#----------------------------------------------------------
def check_args_(opt):
   opt.dtype = 'float'
   interpret_ini_type_(opt)

   #---  Model
   opt.depth = 16; opt.k = 4  # Network depth and width

   #---  Optimization and GULF
   opt.fc_name = 'fc'            # Name of the last FC layer.
   opt.m = -1                    # Do GULF2.
   if opt.num_stages is None:
      opt.num_stages = 15        # Number of stages.
   opt.max_count = 280000        # Length of each stage.  280K steps.
   opt.do_count_epochs = False
   opt.batch_size = 128          # Mini-batch size
   opt.weight_decay = 0.0005     # Weight decay
   opt.lr = 0.01                 # Learning rate
   opt.decay_lr_at = [200000, 240000] # Decay the learning rate after 200K and 240K steps
   opt.lr_decay_ratio = 0.1      # To decay the learning rate, multiply 0.1.

   #---  Data
   opt.dataset = 'SVHN'
   opt.do_download = True # Download data if it does not exist
   opt.do_augment = 0     # 0: Do not augment data.
   opt.ndev = 5000
   opt.dev_seed = 7

   #---  Testing
   opt.do_reduce_testing = True  # Evaluate on the test set only at the end of each stage.
   opt.test_interval = 40000     # Test interval
   opt.inc = 100                 # Interval of showing progress of training.

   #---  Check values and display ...
   raise_if_nonpositive_any(opt, ['depth','k'])
   show_args(opt, ['depth','k','dropout'], 'Model ---')
   raise_if_nonpositive_any(opt, ['num_stages','max_count','batch_size','lr','lr_decay_ratio','alpha'])
   show_args(opt, ['max_count','do_count_epochs', 'batch_size','decay_lr_at','lr_decay_ratio','lr',
                   'weight_decay','num_stages'],
             'Optimization ------')
   if is_gulf(opt):
      show_args(opt, ['ini_type','initial','alpha','m'],
                'GULF training options ------')
   raise_if_nonpositive_any(opt, ['ndev'])
   show_args(opt, ['dataset','dataroot','do_augment','do_download','ndev'], 'Data -----')
   show_args(opt, ['nthread','nthread_test'], '# of workers -----')
   raise_if_nonpositive_any(opt, ['ngpu'])
   show_args(opt, ['seed','ngpu','verbose','save','csv_fn','test_interval'], 'Others ----------')

#********************************************************************
def main():
   parser = ArgParser_HelpWithDefaults(description='train_svhn', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args()
   check_args_(opt)

   resnet_train(opt)

if __name__ == '__main__':
   main()
