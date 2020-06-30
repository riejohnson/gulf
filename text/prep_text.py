import os
import sys
import numpy as np
import argparse
import torch
from utils.utils0 import logging, timeLog, raise_if_None, raise_if_nonpositive, divup, are_these_same
from utils.utils import cast
from .text_utils import download_input_data

PAD_IDX = 0
OOV_TOK = '!OOV!'
Tok_ext = '.tok.txt'
Cat_ext = '.cat'
Vocab_ext = '.vocab.txt'

#   ids: list of token id's, e.g., [ 6,10,300, ], typically for multiple documents.
#   ids_top: list of top positions (in ids) of documents.  The last entry points the end of ids.
#            That is, token id's of the i-th document are ids[ids_top[i]:ids_top[i+1]] for i=0,1,...
#                     and the number of documents is len(ids_top)-1.

#------------------------------------------------------------------
def check_ids(ids, vocab_size):
   if min(ids) < 0 or max(ids) >= vocab_size:
      raise ValueError('id is out of range: min=%d, max=%d, vocab_size=%d' % (min(ids), max(ids), vocab_size))

#------------------------------------------------------------------
def remove_randomly(ids, ids_top, num_max, d_list=None, d_num_max=-1, do_print=False):
   org_d_num = len(ids_top) - 1
   org_num = len(ids)

   if (d_num_max <= 0 or org_d_num <= d_num_max) and (num_max <= 0 or org_num <= num_max):
      return ids, ids_top, d_list

   if do_print:
      timeLog('remove_randomly d_num_max=%d, num_max=%d: (%d,%d) ... ' % (d_num_max, num_max, org_d_num, org_num))

   dxs = torch.randperm(org_d_num).tolist()
   new_ids_top = []
   new_ids = []
   new_d_list = None if d_list is None else []
   for d in dxs:
      new_ids_top += [len(new_ids)]
      new_ids += ids[ids_top[d]:ids_top[d+1]]
      if new_d_list is not None:
         new_d_list += [d_list[d]]

      d_num = len(new_ids_top) - 1
      num = len(new_ids)

      if (d_num_max > 0 and d_num >= d_num_max):
         break
      if (num_max > 0 and num >= num_max):
         break

   new_ids_top += [len(new_ids)]

   if do_print:
      timeLog('end of remove_randomly: (%d,%d)' % (len(new_ids_top)-1, len(new_ids)))

   return tuple(new_ids), tuple(new_ids_top), new_d_list

#------------------------------------------------------------------
def remove_short_ones(ids, ids_top, min_dlen, do_d_list=False, do_print=False):
   d_num = len(ids_top) - 1
   if do_print and min_dlen > 0:
      timeLog('remove_short_ones: d_num=%d, min_dlen=%d ... ' % (d_num,min_dlen))
   new_ids = []
   new_ids_top = []
   d_list = [] if do_d_list else None
   for d in range(d_num):
      dlen = ids_top[d+1] - ids_top[d]
      if dlen < min_dlen:
         continue

      new_ids_top += [ len(new_ids) ]
      new_ids += ids[ids_top[d]:ids_top[d+1]]
      if d_list is not None:
         d_list += [d]

   new_ids_top += [ len(new_ids) ]
   if do_print and min_dlen > 0:
      timeLog('remove_short_ones: new d_num=%d ... ' % (len(new_ids_top)-1))

   return tuple(new_ids), tuple(new_ids_top), d_list

#------------------------------------------------------------------
# shuffle id's according to dxs (a list of document#'s).
def shuffle_ids(dxs, ids, ids_top):
   d_num = len(ids_top) - 1
   new_ids_top = []
   new_ids = []
   for dx in dxs:
      if dx < 0 or dx >= d_num:
         raise IndexError
      new_ids_top += [len(new_ids)]
      new_ids += ids[ids_top[dx]:ids_top[dx+1]]
   new_ids_top += [len(new_ids)]
   return new_ids, new_ids_top

#------------------------------------------------------------------
# Uni-gram token id's of a set of documents.
#------------------------------------------------------------------
class TextData_Uni(object):
   def __init__(self, pathname=None, min_dlen=-1, num_max=-1, d_num_max=-1):
      self._ids = []
      self._ids_top = [0]
      self._vocab = None
      self._d_list = []
      if pathname != None and pathname != '':
         self.load(pathname)
         min_dlen = max(0, min_dlen)
         self._ids,self._ids_top,self._d_list = remove_short_ones(self._ids, self._ids_top, min_dlen,
                                                                  do_d_list=True, do_print=True)
         if num_max > 0 or d_num_max > 0:
            self._ids,self._ids_top,self._d_list = remove_randomly(self._ids, self._ids_top, d_list=self._d_list,
                   num_max=num_max, d_num_max=d_num_max, do_print=True)
         self._d_list = tuple(self._d_list)

   #---  Reorder documents according to dxs (a list of document#'s).
   def shuffle(self, dxs):
      self._ids,self._ids_top = shuffle_ids(dxs, self._ids, self._ids_top)
      self._ids = tuple(self._ids)
      self._ids_top = tuple(self._ids_top)
      self.d_list = []

   def is_same(self,oth):
      return are_these_same(self, oth, ['_ids','_ids_top','_vocab','_d_list'])
   def d_list(self):
      return self._d_list
   def num_datasets(self):
      return 1
   def get_ids(self):
      return self._ids,self._ids_top
   def vocab(self):
      return self._vocab
   def n_min(self):
      return 1
   def n_max(self):
      return 1
   def rev_vocab(self):
      return gen_rev_vocab(self._vocab)

   def save(self, pathname):
      timeLog('Saving data to ' + pathname)
      torch.save(dict(ids=self._ids, ids_top=self._ids_top,
                      vocab = self._vocab),
                 pathname)
      timeLog('Done with saving ...')
   def load(self, pathname):
      timeLog('Loading data from ' + pathname)
      d = torch.load(pathname)
      self._ids = tuple(d['ids'])
      self._ids_top = tuple(d['ids_top'])
      self._vocab = d['vocab']
      check_ids(self._ids, self.vocab_size())
      timeLog('Done with loading ...')

   def create(self, tok_pathname, vocab_pathname, do_oov):
      timeLog('Creating data: tokens in ' + tok_pathname)
      self._vocab = read_vocab(vocab_pathname, do_oov=do_oov)
      self._ids, self._ids_top = tokens_to_ids(tok_pathname, self._vocab, do_oov=do_oov)
      timeLog('Done with creating data ... ')

   def __len__(self):
      return len(self._ids_top)-1
   def vocab_size(self):
      if self._vocab == None:
         return 0
      else:
         return max(self._vocab.values())+1

   def data_len(self, d):
      self._check_index(d)
      return (self._ids_top[d+1] - self._ids_top[d])

   def _check_index(self, d):
      if d < 0 or d >= len(self):
         raise IndexError

   def __getitem__(self, d):
      self._check_index(d)
      return self._ids[self._ids_top[d]:self._ids_top[d+1]]

#------------------------------------------------------------------
#  Labels of a set of data points.  One label per data point.
#------------------------------------------------------------------
class TextData_Lab(object):
   def __init__(self, pathname=None):
      self._labels = []
      self._num_class = 0
      if pathname != None and pathname != '':
         self.load(pathname)

   #---  Reorder lables according to the order of dxs (a list of document#'s).
   def shuffle(self, dxs):
      new_labels = []
      for d in dxs:
         self._check_index(d)
         new_labels += [self._labels[d]]

      self._labels = tuple(new_labels)

   #---
   def save(self, pathname):
      timeLog('Saving data to ' + pathname)
      torch.save(dict(labels=self._labels, num_class=self._num_class),
                 pathname)
      timeLog('Done with saving ...')
   def load(self, pathname):
      timeLog('Loading data from ' + pathname)
      d = torch.load(pathname)
      self._labels = tuple(d['labels'])
      self._num_class = d['num_class']
      timeLog('Done with loading ...')

   def create(self, lab_pathname, catdic_pathname):
      timeLog('Creating data: labels in ' + lab_pathname)
      self._labels,self._num_class = read_labels(lab_pathname, catdic_pathname)

   def __len__(self):
      return len(self._labels)

   def num_class(self):
      return self._num_class

   def _check_index(self, d):
      if d < 0 or d >= len(self):
         raise IndexError

   def __getitem__(self, d):
      self._check_index(d)
      return self._labels[d]

#------------------------------------------------------------------------
def show_uni_data(uni_data, dxs, lab_data=None):
   v_to_k = uni_data.rev_vocab()
   #---
   def get_k(v):
      k = v_to_k[v]
      if k == '':
         k = '[]'
      return k
   #---
   for dx in dxs:
      ids = uni_data[dx]
      lab = '' if lab_data is None else lab_data[dx]
      s = '%d (%s) ----------\n' % (dx,lab)
      for i,id in enumerate(ids):
         s += (get_k(id)+' ')
      logging(s)

#------------------------------------------------------------------------
#  Generate a vocabulary of uni-grams.  The most frequent ones are retained.
#  Output: a text file containing one word per line.
#------------------------------------------------------------------------
def gen_uni_vocab(inp_path, out_path, num_max, do_show=False):
   inc = 100000
   timeLog('gen_uni_vocab -- input:%s, output:%s, num_max=%d' % (inp_path, out_path, num_max))
   with open(inp_path, encoding="utf8") as f:
      docs = f.read().split('\n')

   num = len(docs)
   if docs[-1] == '':
      num -= 1

   vocab = {}
   logging('num=%d' % num)
   for d in range(num):
      tokens = docs[d].lower().split()
      for token in tokens:
         if vocab.get(token) is None:
            vocab[token] = 1
         else:
            vocab[token] += 1

      if do_show and ((d+1)%inc == 0 or d == num-1):
         timeLog('... %d (%d) ... ' % (d+1, len(vocab)))

   logging('size of vocab=%d' % len(vocab))

   #---  sort by count
   vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
   vocab_list = [ item[0]+'\n' for i,item in enumerate(vocab_sorted) if num_max <= 0 or i < num_max ]
   min_count = vocab_sorted[len(vocab_list)-1][1]
   logging('size of vocab=%d, min_count=%d' % (len(vocab_list),min_count))

   with open(out_path, mode='w', encoding="utf8") as f:
      f.writelines(vocab_list)

#------------------------------------------------------------------------
def read_vocab(pathname, do_oov=False):
   with open(pathname, encoding="utf8") as f:
      input = f.read().split('\n')
   vocab_size = len(input)
   if input[-1] == '':
      vocab_size -= 1

   vocab = { input[i].strip().split()[0]: i+1 for i in range(vocab_size) } # '+1' for reserving 0 for padding
   vocab[''] = 0

   if do_oov:
      if OOV_TOK in vocab.keys():
         raise Exception('OOV_TOK exists in text?!')
      vocab[OOV_TOK] = max(vocab.values()) + 1

   logging('vocabulary size = %d' % (max(vocab.values())+1))
   return vocab

#------------------------------------------------------------------------
def gen_rev_vocab(vocab):
   max_value = max(vocab.values())
   v_to_k = [ '' for i in range(0,max_value+1) ]
   for k,v in vocab.items():
     v_to_k[v] = k
   return v_to_k

#------------------------------------------------------------------------
def read_labels(lab_pathname, catdic_pathname):
   with open(lab_pathname) as f:
      labels = f.read().split()
   with open(catdic_pathname) as f:
      list_cat = f.read().split()
      dic_cat = { list_cat[i]: i for i in range(len(list_cat)) }

   return [ dic_cat[labels[i]] for i in range(len(labels)) ], len(list_cat)

#------------------------------------------------------------------------
def tokens_to_ids(pathname, vocab, do_oov=False):
   timeLog('tokens_to_ids ... do_oov='+str(do_oov))
   with open(pathname, encoding="utf8") as f:
      docs = f.read().split('\n')

   num_docs = len(docs)
   if docs[-1] == '':
      num_docs -= 1

   logging('# of doc='+str(num_docs))
   ids_top = [ 0 for i in range(num_docs+1) ]
   num_tokens = 0
   for d in range(num_docs):
      num_tokens += len(docs[d].split())
      ids_top[d+1] = num_tokens
   logging('# of token occurrences ='+str(num_tokens))

   oov_id = 0
   if do_oov:
      oov_id = vocab.get(OOV_TOK)
      if oov_id is None:
         raise Exception('oov_id is None??')

   oov = 0
   ids = [ 0 for i in range(num_tokens) ]
   max_len = 0
   inc = 100000
   for d in range(num_docs):
      if inc > 0 and d % inc == 0:
         timeLog('tokens_to_ids ... %d ... ' % d)

      tokens = docs[d].lower().split()
      max_len = max(max_len, len(tokens))
      if len(tokens) <= 0:
        logging("***WARNING*** empty text: d="+str(d))

      for i in range(len(tokens)):
         index = vocab.get(tokens[i])
         if index != None:
            ids[ids_top[d]+i] = index
         else:
            if do_oov:
               ids[ids_top[d]+i] = oov_id
            oov += 1

      if len(tokens) != ids_top[d+1] - ids_top[d]:
         raise Exception('something is wrong ...')

   logging('avg,max='+str(num_tokens/num_docs)+','+str(max_len))
   logging('#oov='+str(oov)+' ratio='+str(oov/num_tokens))
   return ids, ids_top

#------------------------------------------------------------------------
class TextDataBatches(object):
   def __init__(self, dataset, labset, batch_size, do_shuffle, req_max_len,
                batch_unit=-1, req_min_len=-1,
                x_dss=None):
      assert dataset is not None
      assert labset is not None
      if len(dataset) != len(labset):
         raise Exception('Conflict in sizes of dataset and labset')

      self.ds = dataset   # TextData_Uni
      self.ls = labset    # TextData_Lab
      self.x_dss = x_dss  # list of TextData_N

      self._batches = []  # list of list, e.g., [[10,3,0], [2,5,2], ... ]

      self._batch_unit = batch_unit
      self._batch_size = batch_size
      self._do_shuffle = do_shuffle

      self._dxs = []
      self._dxs_pos = 0
      self._req_min_len = req_min_len
      self._req_max_len = req_max_len

      self._create_dxs()

      if batch_unit > 0:
         self._make_batches()

   def num_batches(self):
      if self._batch_unit > 0:
         return len(self._bxs)
      else:
         return max(1, len(self._dxs)//self.batch_size)

   def _create_dxs(self):
      self._dxs = list(range(len(self.ds)))
      self._dxs_pos = 0

   def __len__(self):
      return len(self.ds)

   def __iter__(self):
      return self

   def __next__(self):
      if self._batch_unit > 0: # text lengths in a batch are roughly the same.  fast.
         if self._bxs_pos >= len(self._bxs):
            self._bxs_pos = 0 # prepare for the next epoch
            raise StopIteration

         if self._bxs_pos == 0 and self._do_shuffle:
            self._make_batches()

         bx = self._bxs[self._bxs_pos]  # batch id
         idx = self._batches[bx]
         self._bxs_pos += 1

      else: # text lengths in a batch vary a lot.  good randomization but slow.
         num = min(self._batch_size, len(self._dxs) - self._dxs_pos)
         if num <= 0:
            self._dxs_pos = 0 # prepare for the next epoch
            raise StopIteration

         if self._dxs_pos == 0 and self._do_shuffle:
            np.random.shuffle(self._dxs)

         idx = [ self._dxs[self._dxs_pos+i] for i in range(num) ]
         self._dxs_pos += self._batch_size

      if self.x_dss == None:
         return self._gen_batch(idx)
      else:
         return self._gen_batch_x(idx)

   def _gen_batch(self, idx): # return [data, labels]
      max_len = self._req_min_len
      for d in idx:
         max_len = max(max_len, self.ds.data_len(d))

      # padding
      data = [ (list(self.ds[d]) + [ PAD_IDX for i in range(max_len - self.ds.data_len(d)) ]) for d in idx ]

      # shortening
      if self._req_max_len > 0 and self._req_max_len < max_len:
         data = [ data[i][0:self._req_max_len] for i in range(len(data)) ]

      data = cast(torch.tensor(data), 'long')
      labels = [ self.ls[d] for d in idx ]
      labels = cast(torch.tensor(labels), 'long')
      return [ data, labels ]

   def _gen_batch_x(self, idx): # return [data, labels, extra data]
      if len(self.x_dss) > 1:
         raise ValueError('no support (yet) for len(x_dss) > 1')

      data_lab = self._gen_batch(idx)  # [data,labels]
      data_len = data_lab[0].size(1)   # length of data

      x_ds = self.x_dss[0]
      # padding
      x_data = [ (list(x_ds[d]) + [ PAD_IDX for i in range(data_len - x_ds.data_len(d)) ]) for d in idx ]
      # shortening
      x_data = [ x_data[i][0:data_len] for i in range(len(x_data)) ]
      x_data = cast(torch.tensor(x_data), 'long')

      return data_lab + [[x_data]]

   #  Sort documents roughly by length.  "unit" is for making it "rough".
   def _make_batches(self):
      eyec = "TextDataBatches::_make_batches"
#      logging("%s making batches ... unit: %d" % (eyec,self._batch_unit))
      unit = self._batch_unit
      dict = {}
      dxs = [ i for i in range(len(self)) ]
      if self._do_shuffle:
         np.random.shuffle(dxs)
      total_len = 0
      for i in range(len(dxs)):
         dx = dxs[i]
         data_len = self.ds.data_len(dx)
         dict[dx] = data_len // unit
         total_len += data_len

      sorted_d = sorted(dict.items(), key=lambda x: x[1])  # Roughly sort by length
      self._dxs = []
      for i in range(len(sorted_d)):
         self._dxs += [ sorted_d[i][0] ]

      num_batches = (len(self._dxs) + self._batch_size - 1) // self._batch_size
      logging("%s #batches=%d, #data=%d, #tokens=%d" % (eyec, num_batches, len(self._dxs), total_len))

      #-----
      self._batches = []
      sz = len(self)

      if self._batch_size > sz:
         raise Exception('batch_size must not exceed the size of dataset: '
                      +str(self._batch_size)+','+str(sz))

      for b in range(num_batches):
         dxs_pos = b*self._batch_size
         num = min(self._batch_size, sz - dxs_pos)
         self._batches += [[ self._dxs[dxs_pos+i] for i in range(num) ]]

      #-----
      self._bxs = [ i for i in range(num_batches) ]
      if self._do_shuffle:
         np.random.shuffle(self._bxs)

      self._bxs_pos = 0

#------------------------------------------------------------------------
#  interface modules
#------------------------------------------------------------------------
def create_uni_lab(input_name, vocab_name, types, output_name, do_oov=True):
   catdic = input_name+'.catdic'

   for type in types: # [ dv,test,td ]
      data_onm = gen_uni_name('', output_name, type)
      lab_onm  = gen_lab_name('', output_name, type)
      tnm = gen_tok_name('', input_name, type) # token file name
      lnm = gen_cat_name('', input_name, type) # cat (label) file name

      data = TextData_Uni()
      data.create(tnm, vocab_name, do_oov)
      data.save(data_onm)

      lab = TextData_Lab()
      lab.create(lnm, catdic)
      lab.save(lab_onm)

#------------------------------------------------------------------------
def create_uni(input_name, vocab_name, output_name, do_oov=True):
   data_onm = output_name + '-uni.pth'
   data = TextData_Uni()
   data.create(input_name, vocab_name, do_oov)
   data.save(data_onm)

#------------------------------------------------------------------------
def compare_vocab(path0, path1):
   vocab0 = read_vocab(path0)
   vocab1 = read_vocab(path1)
   keys0 = vocab0.keys()
   keys1 = vocab1.keys()
   s0 = sorted(vocab0.items(), key=lambda x: x[1])
   s1 = sorted(vocab1.items(), key=lambda x: x[1])

   only_in_v1 = [ (k1,v1) for k1,v1 in s1 if not k1 in keys0 ]
   only_in_v0 = [ (k0,v0) for k0,v0 in s0 if not k0 in keys1 ]

   def show_list(lst, path):
      logging('%d, only in %s --------------' % (len(lst),path))
      for k,v in lst:
         logging('%d: %s' % (v,k))

   show_list(only_in_v1, path1)
   show_list(only_in_v0, path0)

#------------------------------------------------------------------------
def show_uni_lab(uni_path, lab_path, dxs):
   uni_data = TextData_Uni(uni_path)
   lab_data = TextData_Lab(lab_path) if lab_path else None
   show_uni_data(uni_data, dxs, lab_data)

#------------------------------------------------------------------------
#  File naming conventions
#------------------------------------------------------------------------
def where(dataroot):
   if dataroot:
      return dataroot+os.path.sep
   else:
      return ''

def gen_vocab_name(dataroot, dataset, vocab_max):
   return where(dataroot) + dataset + '-max' + str(vocab_max) + Vocab_ext

def gen_tok_name(dataroot, dataset, type):
   return where(dataroot) + dataset + '-' + type + Tok_ext

def gen_cat_name(dataroot, dataset, type):
   return where(dataroot) + dataset + '-' + type + Cat_ext

def gen_uni_name(dataroot, dataset, type):
   return where(dataroot) + dataset + '-' + type + '-uni.pth'

def gen_lab_name(dataroot, dataset, type):
   return where(dataroot) + dataset + '-' + type + '-lab.pth'

#------------------------------------------------------------------------
def main(action, args):
   print('prep_text.main: args=', args)

   parser = argparse.ArgumentParser(description='text_prep.unigram', formatter_class=argparse.MetavarTypeHelpFormatter)

   if action == 'prep':
      parser.add_argument('--dataset', type=str, required=True)
      parser.add_argument('--dataroot', type=str, required=True)
      parser.add_argument('--vocab_max', type=int, default=30000)
      opt = parser.parse_args(args)
      input_name = gen_tok_name(opt.dataroot, opt.dataset, 'train')
      if not os.path.exists(input_name):
         download_input_data(opt.dataroot, opt.dataset, for_what=input_name)
      vocab_name = gen_vocab_name(opt.dataroot, opt.dataset, opt.vocab_max)
      if os.path.exists(vocab_name):
         timeLog('Using existing vocab: %s' % vocab_name)
      else:
         timeLog('Generaing vocab: %s --------------' % vocab_name)
         gen_uni_vocab(input_name, vocab_name, opt.vocab_max, do_show=True)

      do_oov = True
      input_name = opt.dataroot + os.path.sep + opt.dataset
      output_name = input_name
      types = [ 'train', 'test' ]
      create_uni_lab(input_name, vocab_name, types, output_name, do_oov=do_oov)

   elif action == 'gen_uni_vocab':
      parser.add_argument('--input_name', type=str, required=True)
      parser.add_argument('--vocab_max', type=int, default=30000)
      parser.add_argument('--vocab_name', type=str, required=True)
      opt = parser.parse_args(args)
      gen_uni_vocab(opt.input_name, opt.vocab_name, opt.vocab_max, do_show=True)

   elif action == 'create_uni_lab':
      parser.add_argument('--input_name', type=str, required=True)
      parser.add_argument('--vocab_name', type=str, required=True)
      parser.add_argument('--output_name', type=str, required=True)
      parser.add_argument('--types', default=['train', 'test'], type=str, nargs='+')
      opt = parser.parse_args(args)
      create_uni_lab(opt.input_name, opt.vocab_name, opt.types, opt.output_name, do_oov=True)

   elif action == 'create_uni':
      parser.add_argument('--input_name', type=str, required=True)
      parser.add_argument('--vocab_name', type=str, required=True)
      parser.add_argument('--output_name', type=str, required=True)
      opt = parser.parse_args(args)
      create_uni(opt.input_name, opt.vocab_name, opt.output_name, do_oov=True)

   elif action == 'show_data':
      parser.add_argument('--uni_path', type=str, required=True)
      parser.add_argument('--lab_path', type=str)
      parser.add_argument('--dxs', type=int, nargs='+', default=[0,1,2])
      opt = parser.parse_args(args)
      show_uni_lab(opt.uni_path, opt.lab_path, opt.dxs)

   else:
      logging('Unknown action: %s.  Try gen_uni_vocab, create_uni_lab, create_uni, or show_data.' % action)
      return
