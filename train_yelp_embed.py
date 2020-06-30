import sys
import os
import argparse
from utils.utils0 import logging, timeLog, raise_if_nonpositive_any, show_args, ArgParser_HelpWithDefaults
from text.dpcnn_train_embed import main as dpcnn_train_embed

#------------------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--n_max', default=1, type=int, choices=[1,3], help='Specify 3 for training an embedding function of a bag of {1,2,3}-grams.')
   parser.add_argument('--dataroot', default='data', type=str, help='Directory where input data files are.')
   parser.add_argument('--emb_save', type=str, help='Pathname to save the trained embedding.')

   parser.add_argument('--dont_write_to_dataroot', action='store_true', help="Do not write files to the 'dataroot' directory.")
   parser.add_argument('--num_workers', type=int, default=6, help='Number of threads for generating input data.')
   parser.add_argument('--seed', default=1, type=int, help='Random seed.')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')
   parser.add_argument('--ngpu', default=1, type=int, help='number of GPUs.')
   parser.add_argument('--max_grad_norm', default=-1, type=float)

#------------------------------------------------------------------
def check_args_(opt):
   opt.width = 250              # Dimensionality of the embedding output.
   opt.dataset = 'yelppol'
   opt.region = 5 if opt.n_max == 3 else 3

   if opt.emb_save is None:
      dir = 'emb'
      if not os.path.exists(dir):
         os.mkdir(dir)
      opt.emb_save = dir+os.path.sep+opt.dataset+'-n'+str(opt.n_max)+'r'+str(opt.region)+'-emb.pth'

   opt.num_max_each = -1; opt.target_region = -1
   opt.do_target_count = True; opt.dont_erase_zerotarget = False; opt.dont_merge_target = False
   opt.min_dlen = opt.region * 2

   opt.num_stages = 2           # Number of stages for optimization with lr resetting.
   opt.max_count = 1000000      # Length of each stage.
   opt.do_count_epochs = False  #
   opt.do_save_interim = True   #
   opt.batch_size = 100         # Mini-batch size.
   opt.lr = 5.0                 # Learning rate.
   opt.weight_decay = 0         # Weight decay.
   opt.decay_lr_at = [ 800000 ] # When to decay the learning rate.
   opt.lr_decay_ratio = 0.1     # Learning rate decay ratio.
   opt.inc = 1000               # Interval of showing progress of training.

   raise_if_nonpositive_any(opt, ['n_max','n_max','num_workers','width','num_stages','max_count','batch_size','lr','lr_decay_ratio'])
   show_args(opt, ['width','region','min_dlen','target_region','do_target_count','dont_erase_zerotarget','dont_merge_target','emb_save'], 'Embedding ---')
   show_args(opt, ['dataset','dataroot'], 'Input data -----')
   show_args(opt, ['num_stages','max_count','do_count_epochs', 'batch_size','decay_lr_at','lr_decay_ratio','lr','max_grad_norm'],
             'Optimization ------')
   show_args(opt, ['seed','ngpu','num_workers','verbose'], 'Others ---')

#********************************************************************
def main():
   timeLog("train_yelp_embed begins ...")

   parser = ArgParser_HelpWithDefaults(description='train_yelp_embed', formatter_class=argparse.MetavarTypeHelpFormatter)

   add_args_(parser)

   opt = parser.parse_args()
   opt.dtype = 'float'
   check_args_(opt)
#   if opt.verbose:
#      show_args(opt, vars(opt).keys(), header='All of the parsed options: ')

   dpcnn_train_embed(opt)

   timeLog("train_yelp_embed ends ...")

if __name__ == '__main__':
   main()