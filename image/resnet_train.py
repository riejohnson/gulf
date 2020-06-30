import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import cast, data_parallel
from torch.backends import cudnn
from .resnet import resnet

from utils.utils0 import logging, reset_logging, timeLog, raise_if_absent, add_if_absent_
from .data import dataset_attr, create_iterators_tddevtst
from .data import check_opt_for_create_iterators_tddevtst_ as check_opt_for_data_
from gulf import is_gulf, train_base_model, train_gulf_model, copy_params, Target_index

cudnn.benchmark = True

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, [ 'seed','depth','k','dropout','ngpu','dataset','dtype'], who='resnet_train')
   add_if_absent_(opt, ['csv_fn'], '')

#********************************************************************
def main(opt):
   timeLog("resnet_train(opt) begins ...")
   check_opt_(opt)
   check_opt_for_data_(opt)
   
   logging('Using %s ... ' % ('GPU(s)' if torch.cuda.is_available() else 'CPU'))

   reset_logging(opt.csv_fn)

   torch.manual_seed(opt.seed)
   np.random.seed(opt.seed)

   #---  prepare net
   num_classes = dataset_attr(opt.dataset)['nclass']

   def initialize_model():
      return resnet(opt.depth, opt.k, num_classes, dropout=opt.dropout)

   func, params = initialize_model()

   #---  prepare data
   do_pin_memory = torch.cuda.is_available()
   rs = np.random.get_state()
   train_loader, dev_loader, test_loader = create_iterators_tddevtst(opt, do_pin_memory)
   np.random.set_state(rs)
   test_dss = [ {'name':'dev', 'data':dev_loader}, {'name':'test', 'data':test_loader} ]

   #---  training ...
   loss_function = F.cross_entropy
   def net(sample, is_train=False):
      if sample is None:
         return loss_function
      inputs = cast(sample[0], opt.dtype)
      output = data_parallel(func, inputs, params, is_train, list(range(opt.ngpu))).float()
      return loss_function(output, cast(sample[Target_index], 'long')), output

   if not is_gulf(opt):
      train_base_model(opt, net, params, train_loader, test_dss)
   else:
      i_func, i_params = initialize_model()
      copy_params(src=params, dst=i_params)
      def i_net(sample):
         is_train = False
         inputs = cast(sample[0], opt.dtype)
         return data_parallel(i_func, inputs, i_params, is_train, list(range(opt.ngpu))).float()

      train_gulf_model(opt, i_net, i_params, net, params, train_loader, test_dss)

   timeLog("resnet_train(opt) ends ...")
