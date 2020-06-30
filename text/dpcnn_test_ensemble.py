import sys
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchnet as tnt
from utils.utils import cast
from torch.backends import cudnn

from .dpcnn import dpcnn
from .prep_text import TextData_Uni, TextData_Lab, TextDataBatches, gen_uni_name, gen_lab_name
from .prep_text_n import TextData_N, gen_n_name
from utils.utils0 import logging, timeLog, show_args, Clock, raise_if_nonpositive_any, ArgParser_HelpWithDefaults
from utils.utils0 import raise_if_absent, add_if_absent_

from gulf import load, Cdim, Target_index
from .text_utils import load_x_emb, parse_path_with_tags, search_by_id, match_vocab

cudnn.benchmark = True

#----------------------------------------------------------
def parse_model_paths(opt, x_embed):
   models = []
   for path_with_tags in opt.model_paths:
      d = parse_path_with_tags(path_with_tags)
      my_xembed = None
      emb_ids = d.get('emb')
      if emb_ids is not None:
         my_xembed = []
         if not isinstance(emb_ids, list):
            emb_ids = [ emb_ids ]
         for emb_id in emb_ids:
            emb_no = search_by_id(emb_id, x_embed)
            if emb_no < 0:
               raise ValueError('model_paths contains an undefined emb tag: emb_tag=%s, entry=%s' % (emb_id, path_with_tags))
            my_xembed += [ x_embed[emb_no] ]

      models += [ { 'model_path':d['path'], 'x_embed': my_xembed } ]

   return models

#----------------------------------------------------------
def show_model_info(d):
   x_embed = d['x_embed']
   if x_embed is not None:
      for x_emb in x_embed:
         logging('   x_emb: %s' % x_emb['id'])

#----------------------------------------------------------
def test(net, data, opt):
   timeLog('testing ... (%d)' % (len(data)))
   topk = [1]
   inc = opt.test_inc
   sum_loss = 0
   mtr_err = tnt.meter.ClassErrorMeter(topk=topk, accuracy=False)

   test_clk = Clock()
   data_num = 0; count = 0
   for sample in data:
      bsz=sample[0].size(0)
      with torch.no_grad():
         output = net(sample)
      data_num += bsz; count += 1
      mtr_err.add(output.data, sample[Target_index])
      if inc > 0 and count % inc == 0:
         s = '... testing ... ' +str(data_num)+': '+str(float(mtr_err.value()[0]))
         timeLog(s)

   timeLog('... %d ...' % data_num)
   test_clk_tim = test_clk.tick()
   logging('test_clk_tim,%s, num,%d' % (test_clk_tim, data_num))

   return mtr_err.value()

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['model_paths','dataroot','dataset','x_emb','batch_unit','batch_size','depth','width','dropout','top_dropout','ker_size'], 
                   who='dpcnn_test_ensemble')
   add_if_absent_(opt, ['verbose'], False)
   add_if_absent_(opt, ['test_inc'], -1)

#********************************************************************
def main(opt):
   timeLog("dpcnn_test_ensemble(opt) begins ...")
   check_opt_(opt)
   logging('Using %s ... ' % ('GPU(s)' if torch.cuda.is_available() else 'CPU'))

   #---  prepare data
   x_embed,x_n_maxes = load_x_emb(opt.x_emb)

   def prep_uni(type):
      return TextData_Uni(pathname=gen_uni_name(opt.dataroot, opt.dataset, type))
   def prep_lab(type):
      return TextData_Lab(pathname=gen_lab_name(opt.dataroot, opt.dataset, type))
   def prep_x_dss(type): # 'x' for extra
      if x_n_maxes is None:
         return None
      return [ TextData_N(gen_n_name(opt.dataroot, opt.dataset, n_max, type)) for n_max in x_n_maxes ]

   type = 'test'
   ts_ds = prep_uni(type); ts_ls = prep_lab(type); x_ts = prep_x_dss(type) # test data
   if x_ts is not None:
      match_vocab(x_embed, ts_ds, x_ts)

   bch_param = {'req_max_len':opt.req_max_len, 'batch_unit':opt.batch_unit, 'batch_size':opt.batch_size}

   tst_data = TextDataBatches(ts_ds, ts_ls, **bch_param, do_shuffle=False, x_dss=x_ts)

   num_classes = ts_ls.num_class()
   vocab_size = ts_ds.vocab_size()
   logging('#classes=%d, vocab_size=%d' % (num_classes, vocab_size))

   models = parse_model_paths(opt, x_embed)
   for d in models:
      d['func'], d['params'] = dpcnn(opt.depth, opt.width, num_classes, vocab_size,
                      top_dropout=opt.top_dropout, dropout=opt.dropout,
                      ker_size=opt.ker_size,
                      x_embed=d.get('x_embed'))
      load(d['model_path'], d['params'])
      if opt.verbose:
         show_model_info(d)

   timeLog('Ensemble of %d ... ' % len(models))

   def net(sample):
      is_train = False
      inputs = cast(sample[0], 'long')
      x_inputs = [ cast(data, 'long') for data in sample[2] ] if len(sample) >= 3 else None

      sum = 0
      for d in models:
         output = d['func'](inputs, d['params'], is_train, extra_input=x_inputs)
         sum += F.softmax(output, dim=Cdim)
      return sum

   errs = test(net, tst_data, opt)
   s = 'test_err: '
   for err in errs:
      s += ',%.3f' % (err)
   logging(s)

   timeLog("dpcnn_test_ensemble(opt) ends ...")
