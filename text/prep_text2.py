import numpy as np
import argparse
import torch
from utils.utils import cast
from utils.utils0 import logging, timeLog, raise_if_None_any, raise_if_nonpositive_any, raise_if_nonpositive

#------------------------------------------------------------------------
class UnsBow_nomask_Collator(object):
   def __init__(self, num_classes, nosamp_factor=1, do_target_count=False, dont_erase_zerotarget=False):
      logging('UnsBow_nomask_Collator')
      self.num_classes = num_classes
      self.do_target_count = do_target_count
      self.dont_erase_zerotarget = dont_erase_zerotarget
   def __call__(self, data):
      # data: list of dataset[i] where dataset[i] is (bow,neigh) and bow and neigh are a tensor.
      input = torch.cat([ item[0].unsqueeze(0) for item in data ], dim=0).unsqueeze(1) # [#data][#pos][rsz]
      labs = [ item[1] for item in data ]
      targets = gen_targets(labs, self.num_classes, self.do_target_count, self.dont_erase_zerotarget).unsqueeze(-1)
      return (input, targets, None)

#--------------------------------
# labs: list of tensors of fixed length with zero padding
# targets: tensor[#data][#class]
def gen_targets(labs, cnum, do_target_count=False, dont_erase_zerotarget=False):
   targets = torch.zeros((len(labs), cnum), device=torch.device('cpu'))
   if do_target_count:
      for d,d_labs in enumerate(labs):
         targets[d][d_labs] += 1
   else:
      for d,d_labs in enumerate(labs):
         targets[d][d_labs] = 1

   if not dont_erase_zerotarget:
      targets[:,0] = 0  # since dimension '0' is for padding ...

   return targets # cpu tensors

#------------------------------------------------------------------------
#  Produce (input,output) pairs for unsupervised embedding learning.
#------------------------------------------------------------------------
class TextData_UnsBow(object):
   def __init__(self, ds, region, dont_merge_target, target_region=-1):
      self.dont_merge_target = dont_merge_target
      self.ds = ds # TextData_Uni, used for producing both input and output.
      self.region = region
      raise_if_nonpositive(self.region, 'region')
      self.target_region = target_region
      self.setup()

   def num_datasets(self):
      return 1

   def vocab_size(self):
      return self.ds.vocab_size()
   def num_classes(self):
      return self.vocab_size()*2 if self.dont_merge_target else self.vocab_size()

   def check_index(self, index):
      if index < 0 or index >= len(self):
         raise IndexError

   def __len__(self):
      return len(self._to_d)

   def __getitem__(self, index):
      self.check_index(index)
      d = self._to_d[index]; p = index - self._ids_top[d]
      if p < 0 or p >= self._ids_top[d+1]-self._ids_top[d]:
         print('d=d', d, 'p=', p, 'index=', index, 'self._ids_top[d]=', self._ids_top[d], 'self._ids_top[d+1]=', self._ids_top[d+1])
         raise Exception('wrong ...')

      if self.target_region > 0:
         bow,neigh0,neigh1 = gen_data_neigh(self._ids, self._ids_top, d, p, self.region, self.target_region)
      else:
         bow,neigh0,neigh1 = gen_data(self._ids, self._ids_top, d, p, self.region)
      offs = self.vocab_size() if self.dont_merge_target else 0
      return (bow, torch.cat([neigh0,neigh1+offs]))

   #----------------------------
   def setup(self):
      self._ids,self._ids_top = self.ds.get_ids() # just pointing ...
      d_num = len(self._ids_top) - 1
      self._to_d = tuple( [ d for d in range(d_num) for i in range(self._ids_top[d],self._ids_top[d+1]) ] )

      timeLog('TextData_UnsBow::setup (%d) ...' % len(self))

   #----------------------------
   def show_data(self, dxs):
      for dx in dxs:
         logging('%d ----------' % dx)
         bow,neigh = self[dx]
         logging('bow=%s' % str(bow))
         logging('neigh=%s' % str(neigh))


#-----------------------------------------------
def gen_data(ids, ids_top, d, p, region):
   assert region > 0
   half = (region-1)//2
   dtop = ids_top[d]
   dlen = ids_top[d+1] - ids_top[d]

   def gen_one_bow(p):
      beg = dtop + min(dlen, max(0, p-half  ))
      end = dtop + min(dlen, max(0, p+half+1))
      return torch.LongTensor(list(ids[beg:end]) + [ 0 for i in range(region-(end-beg)) ])

   return gen_one_bow(p), gen_one_bow(p-region), gen_one_bow(p+region)

#-----------------------------------------------
def gen_data_neigh(ids, ids_top, d, p, region, tar_region, n_max=1):
   assert tar_region > 0
   dtop = ids_top[d]
   dlen = ids_top[d+1] - ids_top[d]

   def gen_one_bow(p,rsz):
      half = (rsz-1)//2
      beg = dtop + min(dlen, max(0, p-half  ))
      end = dtop + min(dlen, max(0, p+half+1))

      return torch.LongTensor(list(ids[beg:end]) + [ 0 for i in range(rsz-(end-beg)) ])

   n_max_half = 0 if n_max <= 1 else (n_max-1)//2

   dist = (region-1)//2 + (tar_region-1)//2 + 1 + n_max_half
   return gen_one_bow(p,region), gen_one_bow(p-dist,tar_region), gen_one_bow(p+dist,tar_region)


#------------------------------------------------------------------------
from .prep_text import TextData_Uni
#------------------------------------------------------------------------
def main():
   parser = argparse.ArgumentParser(description='')
   parser.add_argument('--action', type=str, required=True)

   #---  for show_uni_lab and show_bow
   parser.add_argument('--uni_pth', type=str)
   parser.add_argument('--region', type=int)
   parser.add_argument('--dxs', type=int, nargs='+')

   opt = parser.parse_args()

   if opt.action == 'show_uns_bow':
      raise_if_None_any(opt, ['uni_pth','dxs'])
      raise_if_nonpositive_any(opt, ['region'])
      uni_ds = TextData_Uni(opt.uni_pth)
      data = TextData_UnsBow(uni_ds, opt.region)
      data.show_data(opt.dxs)
   else:
      raise ValueError('Unknown action: %s' % opt.action)

if __name__ == '__main__':
   main()