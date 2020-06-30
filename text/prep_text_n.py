import os
import sys
import argparse
import torch
from utils.utils0 import logging, timeLog, raise_if_nonpositive, raise_if_None, are_these_same
from .prep_text import gen_rev_vocab, OOV_TOK, shuffle_ids, where, gen_tok_name
from .text_utils import download_input_data

NGRAM_DLM = ' '
do_print = False

def gen_rev_ndict(ndict):
   v_to_k = gen_rev_vocab(ndict)
   return [ item.replace(NGRAM_DLM, "_") for item in v_to_k ]

def find_n_min_max(ndict):
   n_min = 9999
   n_max = 1
   for key in ndict.keys():
      if len(key) > 0 and key != OOV_TOK:
         n = key.count(NGRAM_DLM)+1
         n_min = min(n_min, n)
         n_max = max(n_max, n)
   return n_min,n_max

#------------------------------------------------------------------------
def gen_ngram_vocab(input_path, n_min, n_max, ndict_max, output_path, do_show=False):
   timeLog('gen_ngram_vocab, n_min=%d, n_max=%d, input=%s' % (n_min,n_max,input_path))
   with open(input_path, encoding="utf8") as f:
      docs = f.read().split('\n')

   num = len(docs)
   if docs[-1] == '':
      num -= 1

   ndict = {}
   logging('num=%d' % num)
   for d in range(num):
      tokens = docs[d].lower().split()
      for pos in range(len(tokens)):
         ngram = ''
         for k in range(pos,pos+n_max):
            if k >= len(tokens):
               break
            ngram += NGRAM_DLM if len(ngram) > 0 else ''
            ngram += str(tokens[k])
            if n_min > 0 and k-pos+1 < n_min:
               continue

            if ndict.get(ngram) is None:
               ndict[ngram] = 1
            else:
               ndict[ngram] += 1

      if do_show and ((d+1)%10000 == 0 or d == num-1):
         timeLog('... %d (%d) ... ' % (d+1,len(ndict)))

   logging('size of ndict=%d' % len(ndict))

   #---  sort by count
   ndict_sorted = sorted(ndict.items(), key=lambda x: x[1], reverse=True)
   ndict_list = [ item[0]+'\n' for i,item in enumerate(ndict_sorted) if ndict_max <= 0 or i < ndict_max ]
   min_count = ndict_sorted[len(ndict_list)-1][1]
   logging('size of ndict=%d, min_count=%d' % (len(ndict_list),min_count))

   with open(output_path, mode='w', encoding="utf8") as f:
      f.writelines(ndict_list)

#------------------------------------------------------------------------
def read_ndict(pathname, do_oov=False):
   with open(pathname, encoding="utf8") as f:
      input = f.read().split('\n')
   vocab_size = len(input)
   if input[-1] == '':
      vocab_size -= 1

   vocab = { input[i].strip(): i+1 for i in range(vocab_size) } # '+1' for reserving 0 for padding
   vocab[''] = 0

   if do_oov:
      if OOV_TOK in vocab.keys():
         raise Exception('OOV_TOK exists in text?!')
      vocab[OOV_TOK] = max(vocab.values()) + 1

   logging('ndict size = %d' % (max(vocab.values())+1))

   return vocab

#------------------------------------------------------------------------
# output: ids: list of id's
#         ids_top
def tokens_to_ngram_ids(pathname, ndict, n_min, n_max, do_show=False):
   timeLog('tokens_to_ngram_ids: %s ... ' % pathname)
   with open(pathname, encoding="utf8") as f:
      docs = f.read().split('\n')

   num_docs = len(docs)
   if docs[-1] == '':
      num_docs -= 1

   oov_id = ndict.get(OOV_TOK)
   if oov_id is None:
      oov_id = 0

   logging('# of docs=%d' % num_docs)
   num_tokens = 0
   for d in range(num_docs):
      num_tokens += len(docs[d].lower().split())
   logging('# of token occurrences ='+str(num_tokens))

   ids = []
   ids_top = []
   for d in range(num_docs):
      ids_top += [ len(ids) ]

      tokens = docs[d].lower().split()
      tlen = len(tokens)
      for pos in range(tlen):
         id = oov_id
         n = n_max
         while n >= max(n_min,1):
            ng_f = pos-(n-1)//2
            if ng_f >= 0 and ng_f + n <= tlen:
               ngram = ''
               for k in range(n):
                  if k > 0:
                     ngram += NGRAM_DLM
                  ngram += tokens[ng_f + k]

               my_id = ndict.get(ngram)

               if my_id is not None:
                  id = my_id

            if id != oov_id:
               break

            n -= 1

         ids += [id]

      if do_show and ((d+1)%10000 == 0 or d == num_docs-1):
         timeLog('... %d ... ' % (d+1))

   ids_top += [ len(ids) ]

   if len(ids) != num_tokens or len(ids_top)-1 != num_docs:
      logging('something is wrong: len(ids)=%d, num_tokens=%d, len(ids_top)-1=%d, num_docs=%d'%(len(ids),num_tokens,len(ids_top)-1,num_docs))

   return ids,ids_top

#------------------------------------------------------------------------
def show_ngram_vocab(ndict):
   ndict_v_to_k = gen_rev_ndict(ndict)
   for i,item in enumerate(ndict_v_to_k):
      print(i, ':', item)

#------------------------------------------------------------------
# n-gram id's of a set of documents.
#------------------------------------------------------------------
class TextData_N(object):
   def __init__(self, pathname=None, d_list=None):
      self._ids = []
      self._ids_top = [0]
      self._ndict = None
      self._n_min = 0
      self._n_max = 0
      if pathname is not None:
         self.load(pathname, d_list)

   #---  Reorder documents according to dxs (a list of document#'s).
   def shuffle(self, dxs):
      self._ids,self._ids_top = shuffle_ids(dxs, self._ids, self._ids_top)
      self._ids = tuple(self._ids)
      self._ids_top = tuple(self._ids_top)

   def is_same(self,oth):
      return are_these_same(self, oth, ['_ids','_ids_top','_ndict','_n_min','_n_max'])

   def get_ids(self):
      return self._ids,self._ids_top

   def n_min(self):
      return self._n_min

   def n_max(self):
      return self._n_max

   def __len__(self):
      return len(self._ids_top) - 1

   def _check_index(self, d):
      if d < 0 or d >= len(self):
         raise IndexError

   def data_len(self, d):
      self._check_index(d)
      return (self._ids_top[d+1] - self._ids_top[d])

   def __getitem__(self, d):
      self._check_index(d)
      return self._ids[self._ids_top[d]:self._ids_top[d+1]]

   def create(self, input_pth, ndict, do_show=False):
      self._ndict = ndict
      self._n_min,self._n_max = find_n_min_max(ndict)
      logging('n_min,n_max in ndict = %d,%d' % (self._n_min,self._n_max))
      self._ids, self._ids_top = tokens_to_ngram_ids(input_pth, ndict, self._n_min, self._n_max, do_show)
      self._ids = tuple(self._ids)
      self._ids_top = tuple(self._ids_top)

   def save(self, pathname):
      timeLog('Saving TextData_N to ' + pathname)
      torch.save(dict(ids=self._ids, ids_top=self._ids_top,
                      ndict=self._ndict, n_min=self._n_min, n_max=self._n_max),
                 pathname)
      timeLog('Done with saving ...')

   def load(self, pathname, d_list):
      timeLog('Loading TextData_N from ' + pathname)
      d = torch.load(pathname)
      self._ids = d['ids']
      self._ids_top = d['ids_top']
      self._ndict = d['ndict']
      self._n_max = d['n_max']
      self._n_min = d.get('n_min')
      if self._n_min is None:
         self._n_min = 1

      if d_list is not None:
         logging(' ... keeping %d ... ' % len(d_list))
         new_ids = []
         new_ids_top = []
         for d in d_list:
            new_ids_top += [len(new_ids)]
            new_ids += self._ids[self._ids_top[d]:self._ids_top[d+1]]
         new_ids_top += [len(new_ids)]
         self._ids = new_ids
         self._ids_top = new_ids_top

      self._ids = tuple(self._ids)
      self._ids_top = tuple(self._ids_top)

      timeLog('Done with loading ...')

   def vocab(self):
      return self._ndict

   def vocab_size(self):
      if self._ndict is None:
         return 0
      else:
         return max(self._ndict.values()) + 1

   def show_ndict(self):
      show_ngram_vocab(self._ndict)

   def show_data(self, dxs):
      v_to_k = gen_rev_ndict(self._ndict)
      for dx in dxs:
         logging('%d ------' % dx)
         s = ''
         ids = self[dx]
         for id in ids:
            s += v_to_k[id]
            s += ' '
         logging(s)

#------------------------------------------------------------------------
def ids_to_words(ten, vocab):
   v_to_k = gen_rev_vocab(vocab)
   s = ''
   for v in ten:
      s += v_to_k[v].replace(NGRAM_DLM, "_")
      s += ' '
   return s

#------------------------------------------------------------------------
#  Produce (input,output) pairs for unsuperivsed embedding learning.
#  Essentially, input is represented by a bag of n-grams.
#------------------------------------------------------------------------
class TextData_N_UnsBow(object):
   def __init__(self, ds, tar_ds, n_max, region, dont_merge_target, target_region=-1):
      self.dont_merge_target = dont_merge_target
      self.ds = ds          # TextData_N for producing input.
      self.tar_ds = tar_ds  # TextData_Uni for producing output.
      self.region = region
      self.n_max = n_max
      raise_if_nonpositive(self.region, 'region')
      self.target_region = target_region if target_region > 0 else region
      self.setup()

   def num_datasets(self):
      return 1

   def vocab_size(self):
      return self.ds.vocab_size()

   def num_classes(self):
      return self.tar_ds.vocab_size()*2 if self.dont_merge_target else self.tar_ds.vocab_size()

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

      ids = self._ids; ids_top = self._ids_top
      dtop = self._ids_top[d]
      dlen = ids_top[d+1] - ids_top[d]

      def gen_one_bow(ids, ids_top, p,rsz):
         half = (rsz-1)//2
         beg = dtop + min(dlen, max(0, p-half  ))
         end = dtop + min(dlen, max(0, p+half+1))
         if do_print:
            logging('beg,end=%d,%d'%(beg,end))
         return torch.LongTensor(list(ids[beg:end]) + [ 0 for i in range(rsz-(end-beg)) ])

      bow = gen_one_bow(self._ids, self._ids_top, p, self.region)
      n_max_half = (self.n_max-1)//2
      dist = (self.region-1)//2 + (self.target_region-1)//2 + 1 + n_max_half
      neigh0 = gen_one_bow(self._tar_ids, self._tar_ids_top, p-dist, self.target_region)
      neigh1 = gen_one_bow(self._tar_ids, self._tar_ids_top, p+dist, self.target_region)

      if do_print:
         logging('bow: ' + ids_to_words(bow, self.ds.vocab()))
         logging('neigh0: ' + ids_to_words(neigh0, self.tar_ds.vocab()))
         logging('neigh1: ' + ids_to_words(neigh1, self.tar_ds.vocab()))
         logging(' ')

      offs = self.tar_ds.vocab_size() if self.dont_merge_target else 0
      return (bow, torch.cat([neigh0,neigh1+offs]))

   #----------------------------
   def setup(self):
      self._ids,self._ids_top = self.ds.get_ids() # just pointing ...
      self._tar_ids,self._tar_ids_top = self.tar_ds.get_ids()

      # compare ids_top
      if self._ids_top != self._tar_ids_top:
         raise ValueError('ids_top of ds differs from that of tar_ds.')

      timeLog('TextData_N_UnsBow::setup ids_top check passed ...')

      d_num = len(self._ids_top) - 1
      self._to_d = tuple( [ d for d in range(d_num) for i in range(self._ids_top[d],self._ids_top[d+1]) ] )

      timeLog('TextData_N_UnsBow: done with setup (%d) ...' % len(self))

   #----------------------------
   def show_data(self, dxs):
      for dx in dxs:
         logging('%d ----------' % dx)
         bow,neigh = self[dx]
         logging('bow=%s' % str(bow))
         logging('neigh=%s' % str(neigh))


#------------------------------------------------------------------------
#  Interface modules
#------------------------------------------------------------------------
def create_n(input_name, ndict_name, output_name, n_max, types, do_oov=True):
   ndict = read_ndict(ndict_name, do_oov=do_oov)
   _,n_max_in_ndict = find_n_min_max(ndict)
   if n_max_in_ndict > n_max:
      raise ValueError('n_max=%d was given, but ndict has %d-grams.' % (n_max,n_max_in_ndit))

   if types is None:
      data = TextData_N()
      data.create(input_name, ndict, do_show=True)
      data.save(output_name)
   else:
      for type in types:
         onm = gen_n_name('', output_name, n_max, type)
         tnm = gen_tok_name('', input_name, type)

         data = TextData_N()
         data.create(tnm, ndict, do_show=True)
         data.save(onm)

#------------------
def gen_ndict_name(dataroot, dataset, n_max, ndict_max):
   return where(dataroot) + dataset + ('-n%d'%n_max) + '-max' + str(ndict_max) + '.ndict.txt'

#------------------------------------------------------------------------
def gen_n_name(dataroot, dataset, n_max, type):
   return where(dataroot) + dataset + '-' + type + ('-n%d.pth' % n_max)

#------------------------------------------------------------------------
def main(action, args):
   parser = argparse.ArgumentParser(description='text_prep.ngram', formatter_class=argparse.MetavarTypeHelpFormatter)

   if action == 'prep':
      parser.add_argument('--dataset', type=str, required=True)
      parser.add_argument('--dataroot', type=str, required=True)
      parser.add_argument('--n_max', type=int, default=3)
      parser.add_argument('--ndict_max', type=int, default=200000)

      opt = parser.parse_args(args)
      raise_if_nonpositive(opt.n_max, 'n_max')
      input_name = gen_tok_name(opt.dataroot, opt.dataset, 'train')
      if not os.path.exists(input_name):
         download_input_data(opt.dataroot, opt.dataset, for_what=input_name)

      ndict_name = gen_ndict_name(opt.dataroot, opt.dataset, opt.n_max, opt.ndict_max)
      if os.path.exists(ndict_name):
         logging('Using existing ndict: %s ... ' % ndict_name)
      else:
         do_show = True
         n_min = 1
         gen_ngram_vocab(input_name, n_min, opt.n_max, opt.ndict_max, ndict_name, do_show)

      input_name = opt.dataroot + os.path.sep + opt.dataset
      output_name = input_name
      types = ['train', 'test']
      create_n(input_name, ndict_name, output_name, opt.n_max, types, do_oov=True)

   elif action == 'gen_ndict':
      parser.add_argument('--input_name', type=str, required=True)  # e.g., input_dir/yelp-train.tok.txt
      parser.add_argument('--ndict_name', type=str, required=True)
      parser.add_argument('--n_max', type=int, default=3)
      parser.add_argument('--ndict_max', type=int, default=200000)
      opt = parser.parse_args(args)
      raise_if_nonpositive(opt.n_max, 'n_max')
      do_show = True
      n_min = 1
      gen_ngram_vocab(opt.input_name, n_min, opt.n_max, opt.ndict_max, opt.ndict_name, do_show)
   elif action == 'create':
      parser.add_argument('--input_name', type=str, required=True)  # e.g., input_dir/yelp
      parser.add_argument('--output_name', type=str, required=True) # e.g., output_dir/yelp
      parser.add_argument('--ndict_name', type=str, required=True)
      parser.add_argument('--n_max', type=int, required=True)
      parser.add_argument('--types', type=str, default=['train', 'test'], nargs='+')
      opt = parser.parse_args(args)
      create_n(opt.input_name, opt.ndict_name, opt.output_name, opt.n_max, opt.types, do_oov=True)
   else:
      raise ValueError('invalid action: %s' % action)
