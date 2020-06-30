import sys
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.backends import cudnn
import torchnet as tnt

from torch.utils.data import DataLoader
from .prep_text import TextData_Uni, gen_uni_name
from .prep_text2 import TextData_UnsBow, UnsBow_nomask_Collator
from .prep_text_n import TextData_N, TextData_N_UnsBow, gen_n_name
from .prep_text import main as prep_text_main
from .prep_text_n import main as prep_text_n_main

from utils.utils import cast, data_parallel
from utils.utils0 import logging, timeLog, Clock, raise_if_nonpositive, raise_if_nan, raise_if_negative, Global_state, Local_state, show_args, stem_name, is_writable
from utils.utils0 import raise_if_absent, add_if_absent_
from .dpcnn import region_embedding

from gulf import print_params, create_optim, change_lr_if_needed

cudnn.benchmark = True
Cdim = 1 # dimension for classes

#----------------------------------------------------------
def train(opt, gen_data_loader, o_net, o_params):
   timeLog('train begins ... ')
   clock = Clock()

   def cross_entropy_multi_loss_each(o, sample):
      targets = sample[1].to(o.device)
      targets = targets / (targets.data.sum(dim=Cdim).unsqueeze(dim=Cdim))
      loss = (-1) * (F.log_softmax(o, Cdim) * targets).sum() / sample[1].size(0)
      return loss

   loss_func = cross_entropy_multi_loss_each

   if not is_writable(opt.emb_save):
      raise ValueError('Writability check failed: emb_save=%s' % opt.emb_save)
   logging('%s (emb_save) passed writability check -- a 0-length file was created ... ')

   g_st = Global_state()
   l_st = Local_state()
   vocab = None
   while g_st.lc() < opt.num_stages:
      vocab_info=update(opt, clock, gen_data_loader,
             loss_func,
             o_net, o_params, g_st, l_st)
      l_st.reset()

   emb_save(opt, o_params, vocab_info)

   timeLog('train ends ... ')

class VocabInfo:
   def __init__(self, ds):
      self.vocab = { k:v for k,v in ds.vocab().items() }
      self.n_max = ds.n_max()
class DummyVocabInfo:
   def __init__(self):
      self.vocab = {}; self.n_max = 1

#----------------------------------------------------------
def update(opt, clock, gen_data_loader, loss_func, net, params, g_st, l_st):
   timeLog('update ---------------------------')
   logging('g_st=%s' % str(g_st))
   trn_data = None
   while True:
      _,upd,_ = l_st.get()
      if upd >= opt.max_count:
         break

      trn_data = gen_data_loader()
      _update(opt, clock, loss_func,
              net, params, trn_data, g_st, l_st)

   ds = trn_data.dataset.ds
   vocab_info = VocabInfo(trn_data.dataset.ds)
   if opt.do_save_interim:
      emb_save(opt, params, vocab_info, g_st=g_st)

   epo,upd,_ = l_st.get()
   g_st.update(epo, upd)
   logging('g_st at the end of update: %s' %(str(g_st)))
   return vocab_info

#----------------------------------------------------------
def _update(opt, clock, loss_func, net, params, trn_data, g_st, l_st):
   timeLog('_update')
   max_count = opt.max_count
   epoch,upd,lr_coeff = l_st.get()
   optimizer = create_optim(opt, lr_coeff, params)
   mtr_loss = tnt.meter.AverageValueMeter()

   num_data = 0
   timeLog('epoch ' + str(epoch) + ' upd ' + str(upd))
   for sample in trn_data:
      if upd >= max_count:
         break

      num_data += sample[0].size(0)
      optimizer,lr_coeff = change_lr_if_needed(opt, optimizer, lr_coeff, params, upd=upd)

      output = net(sample, is_train=True)
      loss = loss_func(output, sample)
      mtr_loss.add(float(loss))

      loss.backward()

      if opt.max_grad_norm > 0:
         torch.nn.utils.clip_grad_norm_(params.values(), opt.max_grad_norm)

      optimizer.step(); upd += 1; optimizer.zero_grad()

      #----  show progress
      if opt.inc > 0 and upd % opt.inc == 0:
         timeLog('  ... '+str(upd)+', trn_loss,'+str(mtr_loss.value()[0])+ ',#data,%d' % num_data)
         raise_if_nan(mtr_loss.value()[0])
         mtr_loss.reset()

   epoch += 1
   l_st.reset(epo=epoch, upd=upd, lr_coeff=lr_coeff)

#----------------------------------------------------------
def emb_save(opt, o_params, vocab_info, g_st=None):
   if opt.emb_save == None or opt.emb_save == '':
      return

   stem = stem_name(opt.emb_save, ['-emb.pth', '.pth', '-emb'])
   if g_st is not None:
      fname = stem+'-glc'+str(g_st.lc()+1)+'-emb.pth'
   else:
      fname = stem+'-emb.pth'

   timeLog('Saving (emb): ' + fname + ' ... ')
   torch.save(dict(params=o_params,
                   vocab=vocab_info.vocab if vocab_info is not None else None,
                   n_max=vocab_info.n_max if vocab_info is not None else None,
                   region=opt.region),
              fname)

#------------------------------------------------------------------
def prep_data(opt):
   uni_nm = gen_uni_name(opt.dataroot, opt.dataset, 'train')
   if not os.path.exists(uni_nm):
      if opt.dont_write_to_dataroot:
         raise Exception("%s is missing.  To generate it, turn off 'dont_write_to_dataroot'." % uni_nm)
      timeLog('----------------------------------------------------------------------')
      timeLog('Calling prep_text.main ... ')
      args = [ '--dataroot', opt.dataroot, '--dataset', opt.dataset ]
      # NOTE: This makes *-uni.pth for not only training data but also test data.
      prep_text_main('prep', args)
      timeLog('prep_text.main ended -------------------------------------------------')
   else:
      timeLog('Using existing: %s ... ' % uni_nm)

   n_nm = None
   if opt.n_max is not None and opt.n_max > 1:
      n_nm = gen_n_name(opt.dataroot, opt.dataset, opt.n_max, 'train')
      if not os.path.exists(n_nm):
         if opt.dont_write_to_dataroot:
            raise Exception("%s is missing.  To generate it, turn off 'dont_write_to_dataroot'." % n_nm)
         timeLog('----------------------------------------------------------------------')
         timeLog('Calling prep_text_n.main ... ')
         args = [ '--dataroot', opt.dataroot, '--dataset', opt.dataset, '--n_max', str(opt.n_max) ]
         # NOTE: This makes *-n3.pth (if 3-grams) for not only training data but also test data.
         prep_text_n_main('prep', args)
         timeLog('prep_text_n.main ended -----------------------------------------------')
      else:
         timeLog('Using existing: %s ... ' % n_nm)
   else:
      opt.n_max = 1

   #---------------------------------
   uni = TextData_Uni(pathname=uni_nm)
   num_classes = uni.vocab_size()
   if n_nm is None:
      vocab_size = num_classes
   else:
      ngrams = TextData_N(pathname=n_nm)
      assert ngrams.n_max() == opt.n_max
      vocab_size = ngrams.vocab_size()

   return num_classes,vocab_size, uni_nm, n_nm

#----------------------------------------------------------
def check_opt_(opt):
   raise_if_absent(opt, ['ngpu','num_workers','seed','batch_size','width','dataroot','dataset','emb_save','num_stages','max_count','region','inc'], 
                   who='dpcnn_train_embed')
   add_if_absent_(opt, ['dont_erase_zerotarget','do_target_count','dont_merge_target','verbose','dont_write_to_dataroot','do_save_interim'], False)
   add_if_absent_(opt, ['target_region','num_max_each','min_dlen','max_grad_norm'], -1)

#********************************************************************
def main(opt):
   timeLog("dpcnn_train_embed(opt) begins ...")
   check_opt_(opt)
   logging('Using %s ... ' % ('GPU(s)' if torch.cuda.is_available() else 'CPU'))

   torch.manual_seed(opt.seed)
   np.random.seed(opt.seed)

   num_classes,vocab_size, uni_nm,n_nm = prep_data(opt)
   logging('num_classes=%d, vocab_size=%d, n_max=%d ... ' % (num_classes,vocab_size,opt.n_max))

   #-----
   func, params = region_embedding(opt.width, num_classes, vocab_size)

   #---  display info ...
   if opt.verbose:
      print_params(params)
   n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
   logging('#parameters:' + str(n_parameters))
#   logging({**vars(opt)})

   #--------------------------
   def gen_data_loader():
      td_uni = TextData_Uni(pathname=uni_nm, min_dlen=opt.min_dlen, num_max=opt.num_max_each)

      bow_param = { 'region':opt.region, 'target_region':opt.target_region, 'dont_merge_target':opt.dont_merge_target }

      if n_nm is not None:
         td_n = TextData_N(pathname=n_nm, d_list=td_uni.d_list())
         td_bow = TextData_N_UnsBow(td_n, td_uni, td_n.n_max(), **bow_param)
      else:
         td_bow = TextData_UnsBow(td_uni, **bow_param)

      assert td_bow.num_classes() == num_classes
      assert td_bow.vocab_size() == vocab_size

      collate_fn = UnsBow_nomask_Collator(num_classes=td_bow.num_classes(),
                     do_target_count=opt.do_target_count, dont_erase_zerotarget=opt.dont_erase_zerotarget)

      pin_memory = torch.cuda.is_available()
      bch_param = { 'batch_size':opt.batch_size, 'collate_fn':collate_fn,
                    'pin_memory':pin_memory}

      trn_data = DataLoader(td_bow, **bch_param, shuffle=True, num_workers=opt.num_workers)
      return trn_data
   #--------------------------

   #---  training ...
   def net(sample, is_train):
      inputs = cast(sample[0], 'long')
      output = data_parallel(func, inputs, params, is_train, list(range(opt.ngpu))).float()
      return output

   train(opt, gen_data_loader, net, params)

   timeLog("dpcnn_train_embed(opt) ends ...")
