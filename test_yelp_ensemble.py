import argparse
from utils.utils0 import show_args, raise_if_nonpositive_any, ArgParser_HelpWithDefaults
from text.dpcnn_test_ensemble import main as dpcnn_test_ensemble

#----------------------------------------------------------
def add_args_(parser):
   parser.add_argument('--dataroot', type=str, default='data', help='Root directory of the data.')
   parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='List of model pathnames optionally with embedding tags, '+
                       'e.g., emb=id0:emb=id1:mod/mod0.pth mod/mod1.pth.')
   parser.add_argument('--x_emb', type=str, nargs='+',
                       help='List of external embedding pathnames with id tags, '+
                       'e.g., id0:emb/emb0.pth id1:emb/emb1.pth')
   parser.add_argument('--verbose', action='store_true', help='Display more info.')

#----------------------------------------------------------
def check_args_(opt):
   #!----  these must match with the values used for training ----!#
   opt.ker_size = 3; opt.depth = 7; opt.width = 250
   opt.dropout = 0; opt.top_dropout = 0
   #!-------------------------------------------------------------!#

   opt.dtype = 'float'
   opt.dataset = 'yelppol'
   opt.batch_size = 128
   opt.test_inc = 100
   opt.batch_unit = 32
   opt.req_max_len = -1

   raise_if_nonpositive_any(opt, ['depth','width','batch_size'])
   show_args(opt, ['depth', 'width', 'dropout', 'top_dropout'], 'Model ---')
   show_args(opt, ['batch_size'], 'Testing ------')
   show_args(opt, ['dataset','dataroot',], 'Data -----')
   show_args(opt, ['x_emb'], 'External embeddings -----')
   show_args(opt, ['verbose'], 'Others ---')

#********************************************************************
def main():
   parser = ArgParser_HelpWithDefaults(description='test_yelp_ensemble', formatter_class=argparse.MetavarTypeHelpFormatter)
   add_args_(parser)
   opt = parser.parse_args()
   check_args_(opt)

   dpcnn_test_ensemble(opt)

if __name__ == '__main__':
   main()
