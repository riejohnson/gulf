import os
import torch

import numpy as np
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, ConcatDataset
from utils.utils0 import logging, raise_if_absent

#----------------------------------------------------------
def dataset_attr(ds_name):
   if ds_name == 'CIFAR10':
      nclass=10;   mean=np.array([0.4914,0.4822,0.4465]); sdev=np.array([0.2470,0.2435,0.2616])
   elif ds_name == 'CIFAR100':
      nclass=100;  mean=np.array([0.5071,0.4865,0.4409]); sdev=np.array([0.2673,0.2564,0.2762])
   elif ds_name == 'SVHN':  # 'train' + 'extra' (604388)
      nclass=10;   mean=np.array([0.4309,0.4302,0.4463]); sdev=np.array([0.1965,0.1983,0.1994])
   elif ds_name == 'ImageNet' or ds_name =='imgnet':
      nclass=1000; mean=np.array([0.485, 0.456, 0.406]);  sdev=np.array([0.229, 0.224, 0.225])
   else:
      raise ValueError('CIFAR10/100, SVHN, or ImageNet was expected.  got '+ds_name+' instead ...')

   return { "nclass": nclass, "mean": mean, "sdev": sdev }

#----------------------------------------------------------
def get_ds(ds_name, ds_dir, is_train, opt):
   do_download = opt.do_download

   if ds_name == 'SVHN':
      if is_train:
         train_ds = datasets.SVHN(ds_dir, split='train', download=do_download)
         extra_ds = datasets.SVHN(ds_dir, split='extra', download=do_download)
         return ConcatDataset([train_ds, extra_ds])
      else:
         return datasets.SVHN(ds_dir, split='test', download=do_download)
   elif ds_name == 'CIFAR10' or ds_name == 'CIFAR100':
      return getattr(datasets, ds_name)(ds_dir, train=is_train, download=do_download)
   elif ds_name == 'ImageNet':
      if do_download:
        logging("*** Warning *** Please manually download the Imagenet dataset in advance.")
      return datasets.ImageNet(ds_dir, split='train' if is_train else 'val', download=False)
   else:
      raise ValueError('CIFAR10/100, SVHN, ImageNet, ImageNetSc, or ImageNetSmall was expected.  got '+ds_name+' instead ...')

#----------------------------------------------------------
def get_tr(ds_name, is_train, do_augment, opt):
   attr = dataset_attr(ds_name)
   mean = attr['mean']
   sdev = attr['sdev']

   if ds_name == 'ImageNet':
      if is_train and do_augment:
         # https://github.com/pytorch/examples/blob/master/imagenet/main.py
         tr = T.Compose([ T.RandomResizedCrop(224),
                          T.RandomHorizontalFlip() ])
      else:
         if is_train:
            logging('!!!!!----- no augment -----!!!!!')
         tr = T.Compose([ T.Resize(256),
                          T.CenterCrop(224) ])  # or TenCrop(224)
   #---  CIFAR10/100 and SVHN
   else:
     tr = T.Compose([ ])
     if do_augment:
        tr = T.Compose([
              tr,
              T.Pad(4, padding_mode='reflect'),
              T.RandomHorizontalFlip(),
              T.RandomCrop(32),
        ])

   return T.Compose([ tr, T.ToTensor(), T.Normalize(mean, sdev) ])

#----------------------------------------------------------
def check_opt_for_create_iterators_tddevtst_(opt):
   raise_if_absent(opt,['dataset','dataroot','do_download','do_augment','nthread','nthread_test','ndev','dev_seed','batch_size'], 
                   who='data.create_iterators_tddevtst')

#----------------------------------------------------------
def create_iterators_tddevtst(opt, do_pin):
   ds_name = opt.dataset; ds_dir = opt.dataroot; do_augment = opt.do_augment
   bs = opt.batch_size; trn_workers = opt.nthread
   tst_workers = max(0, opt.nthread_test)
   dev_num = opt.ndev; dev_seed = opt.dev_seed

   is_train = True
   trn_ds = get_ds(ds_name, ds_dir, is_train, opt)
   trn_num = len(trn_ds)
   trn_idx = list(range(trn_num))
   if dev_num <= 0 or dev_num >= trn_num:
      raise ValueError("dev size must be smaller than the training data size. #train="
                       + str(trn_num) + " requested #dev=" + str(dev_num))

   if dev_seed > 0:
      np.random.seed(dev_seed)
      np.random.shuffle(trn_idx)

   td_idx = trn_idx[dev_num:]
   dev_idx = trn_idx[:dev_num]

   is_train = True
   td_tds  = TransDataset(torch.utils.data.Subset(trn_ds, td_idx),  transform=get_tr(ds_name, is_train, do_augment, opt))
   is_train = False
   dev_tds = TransDataset(torch.utils.data.Subset(trn_ds, dev_idx), transform=get_tr(ds_name, is_train, False,     opt))
   tst_tds = TransDataset(get_ds(ds_name, ds_dir, is_train, opt),   transform=get_tr(ds_name, is_train, False,     opt))

   logging('Data statistics -----')
   logging( '  #train = ' + str(len(td_tds)) + '\n  #dev = ' + str(len(dev_tds)) + '\n  #test = ' + str(len(tst_tds)) )

   return (DataLoader(td_tds,  bs, shuffle=True, drop_last=True, num_workers=trn_workers, pin_memory=do_pin),
           DataLoader(dev_tds, bs, shuffle=False,                num_workers=tst_workers, pin_memory=do_pin),
           DataLoader(tst_tds, bs, shuffle=False,                num_workers=tst_workers, pin_memory=do_pin))

#----------------------------------------------------------
class TransDataset():
   def __init__(self, ds, transform=None, target_tr=None):
      self.ds = ds
      self.transform = transform
      self.target_tr = target_tr
   def __len__(self):
      return len(self.ds)
   def __getitem__(self, index):
      data, target = self.ds[index]
      if self.transform is not None:
         data = self.transform(data)
      if self.target_tr is not None:
         target = self.target_tr(target)

      return data,target
