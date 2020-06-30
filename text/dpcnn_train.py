import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

from utils.utils import cast
from utils.utils0 import logging, reset_logging, timeLog, raise_if_absent, add_if_absent_
from .dpcnn import dpcnn
from .prep_text import TextData_Uni, TextData_Lab, TextDataBatches, gen_uni_name, gen_lab_name
from .prep_text_n import TextData_N, gen_n_name
from .prep_text import main as prep_text_main
from .text_utils import get_dlist, load_x_emb, match_vocab

from gulf import is_gulf, train_base_model, train_gulf_model, copy_params, Target_index

cudnn.benchmark = True

#--------------------------------------------------------------------
#  For a dataset "dataname", the following input files are required.
#     dataname-train.tok.txt, dataname-train.cat
#     dataname-test.tok.txt,  dataname-test.cat
#     dataname.catdic
#
#  *.tok.txt: tokens delimited by white space.  one document per line.
#  *.cat: class labels.
#  *.catdic: class names used in *.cat.  one name per line.
#--------------------------------------------------------------------
#--------------------------------------------------------------------
def prep_data(opt, types):
   missing = ''
   for type in types:
      ds_path = gen_uni_name(opt.dataroot, opt.dataset, type)
      ls_path = gen_lab_name(opt.dataroot, opt.dataset, type)
      if not os.path.exists(ds_path):
         missing += ' %s' % ds_path
      if not os.path.exists(ls_path):
         missing += ' %s' % ls_path
   if len(missing) > 0:
      timeLog('----------------------------------------------------------------------')
      if opt.dont_write_to_dataroot:
         raise Exception("The following files are missing: %s\nTo generate them, turn off 'dont_write_to_dataroot'." % missing)
      timeLog('Calling prep_text_main for creating the following files: %s' % missing)
      prep_text_main('prep', ['--dataset', opt.dataset, '--dataroot', opt.dataroot ])
      timeLog('Done with prep text main ------------------------------')
   else:
      timeLog('Using existing data files ... ')

#----------------------------------------------------------
def check_opt_(opt):
   #---  required attributes 
   names = [ 'dataroot','dataset','num_dev','x_emb','seed','batch_unit','batch_size','depth','width','dropout','top_dropout','ker_size']
   raise_if_absent(opt, names, who='dpcnn_train')

   #---  optional attributes
   add_if_absent_(opt, ['dont_write_to_dataroot'], False)
   add_if_absent_(opt, ['num_train','req_max_len'], -1)
   add_if_absent_(opt, ['train_dlist_path','dev_dlist_path'], None)
   add_if_absent_(opt, ['csv_fn'], '')

#********************************************************************
def main(opt):
   timeLog("dpcnn_train(opt) begins ...")
   check_opt_(opt)
   logging('Using %s ... ' % ('GPU(s)' if torch.cuda.is_available() else 'CPU'))

   reset_logging(opt.csv_fn)

   torch.manual_seed(opt.seed)
   np.random.seed(opt.seed)

   #---  load external embeddings
   x_embed,x_n_maxes = load_x_emb(opt.x_emb)

   #---  prepare data
   prep_data(opt, ['train', 'test'])
   rs = np.random.get_state()

   def prep_uni(type):
      return TextData_Uni(pathname=gen_uni_name(opt.dataroot, opt.dataset, type))
   def prep_lab(type):
      return TextData_Lab(pathname=gen_lab_name(opt.dataroot, opt.dataset, type))
   def prep_x_dss(type): # 'x' for extra
      if x_n_maxes is None:
         return None
      return [ TextData_N(gen_n_name(opt.dataroot, opt.dataset, n_max, type)) for n_max in x_n_maxes ]

   trn_dlist,dev_dlist = get_dlist(opt.seed, opt.num_train, opt.num_dev, opt.train_dlist_path, opt.dev_dlist_path,
                                   len(prep_uni('train')))

   def read_ds_ls_x(dlist): # read data and labels
      type='train'
      ds = prep_uni(type); ds.shuffle(dlist)
      ls = prep_lab(type); ls.shuffle(dlist)
      x_dss = prep_x_dss(type)
      if x_dss is not None:
         for xds in x_dss:
            xds.shuffle(dlist)
      return ds, ls, x_dss

   td_ds,td_ls,x_td = read_ds_ls_x(trn_dlist) # training data
   dv_ds,dv_ls,x_dv = read_ds_ls_x(dev_dlist) # validation data
   match_vocab(x_embed, td_ds, x_td)
   type = 'test'
   ts_ds = prep_uni(type); ts_ls = prep_lab(type); x_ts = prep_x_dss(type) # test data

   bch_param = {'req_max_len':opt.req_max_len, 'batch_unit':opt.batch_unit, 'batch_size':opt.batch_size}

   trn_data = TextDataBatches(td_ds, td_ls, **bch_param, do_shuffle=True,  x_dss=x_td)
   dev_data = TextDataBatches(dv_ds, dv_ls, **bch_param, do_shuffle=False, x_dss=x_dv)
   tst_data = TextDataBatches(ts_ds, ts_ls, **bch_param, do_shuffle=False, x_dss=x_ts)

   np.random.set_state(rs)

   test_dss = [ {'name':'dev', 'data':dev_data}, {'name':'test', 'data':tst_data} ]

   num_classes = td_ls.num_class()
   logging('#classes=%d' % num_classes)
   if num_classes != dv_ls.num_class() or num_classes != ts_ls.num_class():
      raise Exception('Conflict in # of classes: ' +str(num_classes)+','+str(dv_ls.num_class())+','+str(ts_ls.num_class()))
   vocab_size = td_ds.vocab_size()
   logging('#vocab=%d' % vocab_size)
   if vocab_size != dv_ds.vocab_size() or vocab_size != ts_ds.vocab_size():
      raise Exception('Conflict in vocabulary sizes: '+str(vocab_size)+','+str(dv_ds.vocab_size())+','+str(ts_ds.vocab_size()))

   #---  prepare a model
   def initialize_model():
      return dpcnn(opt.depth, opt.width, num_classes, vocab_size,
                   top_dropout=opt.top_dropout, dropout=opt.dropout,
                   ker_size=opt.ker_size,
                   x_embed=x_embed) # external embedding

   func, params = initialize_model()

   #---  training ...
   loss_function = F.cross_entropy
   def net(sample, is_train=False):
      if sample is None:
         return loss_function
      inputs = cast(sample[0], 'long')
      x_inputs = [ cast(data, 'long') for data in sample[2] ] if len(sample) >= 3 else None
      output = func(inputs, params, is_train, extra_input=x_inputs)
      targets = cast(sample[Target_index], 'long')
      return loss_function(output, targets), output

   if not is_gulf(opt):
      train_base_model(opt, net, params, trn_data, test_dss)
   else:
      i_func, i_params = initialize_model()
      copy_params(src=params, dst=i_params)

      def i_net(sample):
         is_train = False
         inputs = cast(sample[0], 'long')
         x_inputs = [ cast(data, 'long') for data in sample[2] ] if len(sample) >= 3 else None
         return i_func(inputs, i_params, is_train, extra_input=x_inputs)

      train_gulf_model(opt, i_net, i_params, net, params, trn_data, test_dss)

   timeLog("dpcnn_train(opt) ends ...")
