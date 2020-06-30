import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from utils.utils0 import logging, timeLog


data_repo = 'http://riejohnson.com/data'
filenames = { 'yelppol': 'yelppol-input-data.tar.gz' }

#------------------------------------------------------------------
def download_input_data(dataroot, dataset, for_what):
   filename = filenames.get(dataset)
   if filename is None:
      raise Exception('%s is not available for downloading.  Please generate it from the original text data.' % for_what)
   else:
      url = data_repo + '/' + filename
      download_and_extract_archive(url, download_root=dataroot, extract_root=dataroot, filename=filename)

#------------------------------------------------------------------
def search_by_id(search_id, searched):
   if searched is None:
      return -1
   for no,d in enumerate(searched):
      id = d.get('id')
      if id == search_id:
        return no
   return -1

#------------------------------------------------------------------
# Verify if a vocabulary of an embedding (x_emb) matches with
#           a vocabulary in a dataset (uni_ds or x_dataset[dsno]).
#------------------------------------------------------------------
def match_vocab(x_emb, uni_ds, x_dataset):
   if x_emb is None:
      return
   for xemb in x_emb:
      xemb_vocab = xemb.get('vocab')
      if xemb_vocab is None: # vocabulary was not saved.
         logging('!WARNING! No vocabulary was saved in: %s' % xemb['path'])
         continue

      dsno = xemb['x_dsno']
      if dsno < 0:
         vocab = uni_ds.vocab()
      else:
         vocab = x_dataset[dsno].vocab()

      if xemb_vocab != vocab:
         raise ValueError('Vocabulary matching failed!: %s, length=%d,%d' % (xemb['path'], len(xemb_vocab), len(vocab)))
      logging('Vocabulary matching was successful: %s' % xemb['path'])

#------------------------------------------------------------------
def parse_path_with_tags(input):
   dlm = ':'
   parts = input.split(dlm)
   num = len(parts)

   d = {}
   d['path'] = parts[num-1]
   for i in range(num-1):
     subparts = parts[i].split('=')
     subnum = len(subparts)
     if subnum == 1 and i == 0:
        d['id'] = subparts[0]
     elif subnum == 2:
        key = subparts[0]; val = subparts[1]
        prev = d.get(key)
        if prev is None:
           d[key] = val
        else:
           if isinstance(prev, list):
              d[key] = prev + [val]
           else:
              d[key] = [prev] + [val]
     else:
        raise ValueError('Invalid tag: tag=%s, entry=%s' % (parts[i], input))

   return d

#------------------------------------------------------------------
def load_emb(xemb):
   timeLog('Loading (emb): %s' % xemb['path'])

   d = torch.load(xemb['path'], map_location='cuda' if torch.cuda.is_available() else 'cpu')
   for k,v in xemb.items():
      if d.get(k) is None: # ignore the tag if the info was saved in the embedding file.
         d[k] = v

   if xemb.get('n_max') is not None and d['n_max'] == xemb['n_max']:
      d['n_max'] = int(xemb['n_max'])

   s = ''
   if d.get('id') is not None:
      s += '   id=%s\n' % d.get('id')
   if d.get('n_max') is not None:
      s += '   n_max=%d\n' % d['n_max']
   s += '   region=%d\n' % d['region']
   if d.get('vocab') is not None:
      s += '   vocab_size=%d\n' % len(d['vocab'])
   logging(s)

   return d


#------------------------------------------------------------------
def load_x_emb(x_emb):
   if x_emb is None or len(x_emb) == 0:
      return None, None

   x_embed = [ parse_path_with_tags(xemb) for xemb in x_emb ]
   x_embed = [ load_emb(xemb) for xemb in x_embed ]

   #---  match the 'n_max' tags in x_embed with integers in x_n_maxes.
   n_maxes = []
   for i,xemb in enumerate(x_embed):
      n_max = xemb.get('n_max')
      if n_max is None:
         continue
      if isinstance(n_max, list):
         raise ValueError('An entry of x_emb should not have more than one n_max tag: %s' % x_emb[i])
      if n_max > 1:
         n_maxes += [n_max]
   n_maxes = [ n_max for n_max in set(n_maxes) ]

   print('n_maxes=', n_maxes)

   #---  match the 'n_max' tags in x_embed with integers in x_n_maxes.
   for i,xemb in enumerate(x_embed):
      dsno = -1
      n_max = xemb.get('n_max')
      if n_max is not None and n_max > 1:
         dsno = n_maxes.index(n_max)

      x_embed[i]['x_dsno'] = dsno

   return x_embed, n_maxes

#------------------------------------------------------------------
def get_dlist(seed,  num_train, num_dev, train_dlist_path, dev_dlist_path, num_data):
   if num_dev > 0:
      if num_train <= 0:
         num_train = max(1, num_data - num_dev)
      
      rs = np.random.get_state()
      np.random.seed(seed)
      if num_train + num_dev > num_data:
         raise ValueError('num_train + num_dev must be no greater than %d.' % num_data)

      indexes = [ d for d in range(num_data) ]
      np.random.shuffle(indexes)
      trn_dlist = indexes[0:num_train]
      dev_dlist = indexes[num_train:num_train+num_dev]
      np.random.set_state(rs)

   elif train_dlist_path and dev_dlist_path:
      def read_dlist(path):
         with open(path, encoding="utf8") as f:
            input = f.read().split('\n')
         dlist = [ int(inp) for inp in input if len(inp) > 0 ]
         logging('read_dlist: size = %d' % len(dlist))
         return dlist

      trn_dlist = read_dlist(train_dlist_path)
      dev_dlist = read_dlist(dev_dlist_path)
   else:
      raise ValueError('Either num_dev/num_train or train_dlist_path/dev_dlist_path is needed.')

   return trn_dlist, dev_dlist
