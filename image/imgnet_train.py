import sys
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import cast
from torch.backends import cudnn
import torchvision.models

from utils.utils0 import logging, reset_logging, timeLog, add_if_absent_, raise_if_absent
from .data import create_iterators_tddevtst
from .data import check_opt_for_create_iterators_tddevtst_ as check_opt_for_data_
from gulf import copy_params, is_gulf, train_base_model, train_gulf_model, Target_index

cudnn.benchmark = True

#----------------------------------------------------------
def get_params(mod):
   params = { n: p for n,p in mod.named_parameters() }
   buffers = { n: p for n,p in mod.named_buffers() }
   params.update(buffers)
   return params

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, [ 'seed','model','ngpu','dataset','dtype'], who='imgnet_train')
   add_if_absent_(opt, ['csv_fn'], '')

#********************************************************************
def main(opt):
   timeLog("imgnet_train begins ...")
   check_opt_(opt)
   check_opt_for_data_(opt)

   reset_logging(opt.csv_fn)

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   logging('Using %s ... ' % ('GPU(s)' if torch.cuda.is_available() else 'CPU'))

   torch.manual_seed(opt.seed)
   np.random.seed(opt.seed)

   #---  prepare net
   def initialize_model():
      if opt.model == 'resnet50pre':
         model = torchvision.models.resnet50(pretrained=True)
      elif opt.model == 'wrn50-2pre':
         model = torchvision.models.wide_resnet50_2(pretrained=True)
      else:
         raise Exception('Invalid model type (for ImageNet): '+opt.model)

      model.to(device)

      params = get_params(model) # names would get a prefix 'module.' if we did this after DataParallel.
      if opt.ngpu > 1:
         model = torch.nn.DataParallel(model)

      return model,params

   model,params = initialize_model()

   #---  prepare data
   assert(opt.dataset == 'ImageNet')
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
      if is_train:
         model.train()
      else:
         model.eval()
      output = model(inputs)
      targets = cast(sample[Target_index], 'long')
      return loss_function(output, targets), output

   if not is_gulf(opt):
      train_base_model(opt, net, params, train_loader, test_dss)
   else:
      i_model,i_params = initialize_model()
      logging('Copying params to i_params')
      copy_params(src=params, dst=i_params)
      def i_net(sample):
         is_train = False
         inputs = cast(sample[0], opt.dtype)
         if is_train:
            i_model.train()
         else:
            i_model.eval()
         return i_model(inputs)

      train_gulf_model(opt, i_net, i_params, net, params, train_loader, test_dss)

   timeLog("imgnet_train ends ...")
