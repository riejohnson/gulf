import argparse
from utils.utils0 import raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from gulf import is_gulf, interpret_ini_type_
from text.dpcnn_train import main as dpcnn_train

#----------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--large_or_small', default='large', type=str, choices=['large','small'],
                       help="'large': use 555K training data points, 'small': use 45K training data points.")
   parser.add_argument('--dataroot', default='data', type=str, help='Root directory of the data.')
   parser.add_argument('--dont_write_to_dataroot', action='store_true', help="Don't write to the 'dataroot' directory.")
   parser.add_argument('--x_emb', type=str, nargs='+', help="Pathnames to external embedding files.")  # external embedding files
   parser.add_argument('--ini_type', default='iniRand', type=str, choices=['iniRand','iniBase','iniBase/2','file','file/2'],
                       help='Parameter initialization method.')
   parser.add_argument('--initial', default='', type=str, help='Pathname of the initial model.  Use this when ini_type==file or file/2.')
   parser.add_argument('--alpha', type=float, help='Alpha.')   
   parser.add_argument('--num_stages', type=int, help='Number of stages.')
   parser.add_argument('--max_grad_norm', default=-1, type=float, help='Maximum gradient norm for gradient clipping. -1: No clipping.')
   parser.add_argument('--seed', default=1, type=int, help='Random seed.')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')
   parser.add_argument('--save', default='', type=str, help='Pathname for saving models.')
   parser.add_argument('--csv_fn', default='', type=str, help='Pathname for writing test results in the CSV format.')

#----------------------------------------------------------
def check_args_(opt):
   opt.dtype = 'float'
   interpret_ini_type_(opt)

   #---  input data
   opt.dataset = 'yelppol'
   opt.num_dev = 5000
   opt.num_train = 555000 if opt.large_or_small == 'large' else 45000
   opt.batch_unit = 32
   opt.req_max_len = -1
   opt.train_dlist_path = opt.dev_dlist_path = None

   #---  model
   opt.depth = 7    # Number of convolutional blocks
   opt.width = 250  # Dimensionality of a convolutional layer.
   opt.dropout = 0; opt.top_dropout = 0  # Dropout
   opt.ker_size = 3 # Kernel size of convolutional layers.

   #---  optimization
   if opt.large_or_small == 'large':
      opt.batch_size = 128
      opt.weight_decay = 0.0001
      opt.lr = 0.25 if opt.x_emb is None else 0.1
   else:
      opt.batch_size = 32
      if opt.x_emb is None:
         opt.weight_decay = 2e-4; opt.lr = 0.25
      else:
         opt.weight_decay = 5e-4; opt.lr = 0.05

   opt.max_count = 10            # Length of each stage.  10 epochs
   opt.do_count_epochs = True
   opt.decay_lr_at = [9]         # Decay the learning rate after 9 epochs
   opt.lr_decay_ratio = 0.1      # To decay the learning rate, multiply 0.1.

   #---  GULF
   if opt.alpha is None:
      if opt.large_or_small == 'large':
         opt.alpha = 0.5
      else:
         opt.alpha = 0.5 if opt.x_emb is not None and opt.ini_type == 'iniRand' else 0.3

   opt.fc_name = 'fc'  # Name of the final FC layer.
   if opt.num_stages is None:
      opt.num_stages = 25 # Number of stages.
   opt.m = -1          # If positive, do GULF1.

   #---  saving and resuming
   opt.resume = ''
   opt.do_noow_save = True       # Save models at the end of each stage without overwriting.
                                 # We need this for making an ensemble later.

   #---  Testing
   opt.do_reduce_testing = True  # Evaluate on the test set only at the end of each stage.
   opt.test_inc = 1000           # Interval of showing progress of testing
   opt.test_interval = 1         # Test interval
   opt.inc = 1000                # Interval of showing progress of training

   #---  Check values and display ...
   raise_if_nonpositive_any(opt, ['ker_size','width','depth'])
   show_args(opt, ['depth', 'width', 'dropout', 'top_dropout'], 'Model ---')
   if opt.train_dlist_path is None:
      raise_if_nonpositive_any(opt, ['num_train','num_dev'])
   show_args(opt, ['dataset','dataroot','dont_write_to_dataroot','large_or_small','num_train','num_dev'], 'Data -----')
   show_args(opt, ['x_emb'], 'External embeddings -----')
   raise_if_nonpositive_any(opt, ['alpha'])
   if is_gulf(opt):
      show_args(opt, ['ini_type','alpha','m','initial'], 'GULF ------')
   else:
      show_args(opt, [], 'Regular training ----')
   raise_if_nonpositive_any(opt, ['num_stages','max_count','lr_decay_ratio','lr','batch_size'])
   show_args(opt, ['num_stages','max_count','do_count_epochs', 'batch_size','decay_lr_at','lr_decay_ratio','lr',
                   'weight_decay','max_grad_norm'], 'Optimization ------')
   show_args(opt, ['save','resume','csv_fn','test_interval','seed','verbose'], 'Others  ------')

#********************************************************************
def main():
   parser = ArgParser_HelpWithDefaults(description='train_yelp', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args()
   check_args_(opt)

   dpcnn_train(opt)

if __name__ == '__main__':
   main()
