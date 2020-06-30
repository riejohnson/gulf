import torch
import torch.nn.functional as F
import utils.utils as utils
from torch.nn.init import normal_

Fdim = 1 # features
Ldim = 2 # length

def resnorm(o):
   return o/(((o**2).sum(dim=Fdim, keepdim=True)+1).sqrt())

def embed(input, params):
   if input.dim() == 2:  # [#data][#pos]
      o = F.embedding(input, params['emb.weight'], padding_idx=0)
   elif input.dim() == 3:  # [#data][#pos][rsz]
      sz = input.size()
      o = F.embedding_bag(input=input.view(-1,sz[2]), weight=params['emb.weight'], mode='sum').view(sz[0],sz[1],-1)

   bias = params.get('emb.b')
   if bias is not None:
      o = o + bias

   o = o.permute(0, 2, 1)

   return o

def embed_plus(input, params):
   o = embed(input, params)
   return resnorm(F.relu(o, inplace=True))

def linear_params(ni, no, std=0.01):
   return {'weight': normal_(torch.Tensor(no, ni), std=std), 'bias': torch.zeros(no)}

#----------------------------------------------------------
def dpcnn(num_block, width, num_classes, vocab_size,
          dropout=0, top_dropout=0, ker_size=3, pool_padding=1,
          x_embed=None):

   if x_embed is not None and len(x_embed) <= 0:
      x_embed = None

   def conv1d_params(ni, no, k, std=0.01):
      p = {'weight': normal_(torch.Tensor(no, ni, k), std=std)}
      return p

   def embedding_params(vocab_size, out_dim, std=0.01):
      p = {'weight': normal_(torch.Tensor(vocab_size, out_dim), std=std)}
      p['weight'][0,:].fill_(0)
      return p

   def embconv_params(ni, no, std=0.01):
      k=1
      p = {'weight': normal_(torch.Tensor(no, ni, k), std=std)}
      p['b'] = torch.zeros(no)
      return p

   def gen_embconv_params(no):
      prms = {}
      for i in range(len(x_embed)):
         ni = x_embed[i]['params']['emb.weight'].size(1)
         prms['conv%d'%i] = embconv_params(ni, no)
      return prms

   def gen_block_params(ni, no):
      return {
         'conv0': conv1d_params(ni, no, ker_size),
         'conv1': conv1d_params(no, no, ker_size),
      }

   def gen_group_params(ni, no, count):
      return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
              for i in range(count)}

   my_params = {
                    'emb': embedding_params(vocab_size, width),
                    'group': gen_group_params(width, width, num_block),
                    'fc': linear_params(width, num_classes)
               }

   if x_embed is not None:
      my_params['embconv'] = gen_embconv_params(width)

   flat_params = utils.cast(utils.flatten(my_params))
   utils.set_requires_grad_except_bn_(flat_params)

   def block(x, params, base, mode, do_downsample, do_skip_1stAct=False):
      o = x
      if not do_skip_1stAct:
         o = F.relu(o, inplace=True)

      o = F.conv1d(o, params[base+'.conv0.weight'], stride=1, padding=int((ker_size-1)/2))
      o = F.relu(o, inplace=True)

      if dropout != None and dropout > 0:
         o = F.dropout(o, dropout, mode)

      o = F.conv1d(o, params[base+'.conv1.weight'], stride=1, padding=int((ker_size-1)/2))
      o = o + x
      if do_downsample:
         if pool_padding > 0 or o.size()[-1] > 2:
            o = F._max_pool1d(o, 3, stride=2, padding=pool_padding)

      return o

   def gen_bow(input, rsz, for_seq=False):
      sz0 = input.size(0); sz1 = input.size(1)
      half = (rsz-1)//2
      seq = []
      for r in range(rsz):
         shift = r - half
         if shift < 0:
            seq += [torch.cat( [input[:,-shift:sz1], torch.zeros(sz0, -shift, device=input.device).long()], dim=1 ).unsqueeze(-1)]
         elif shift >= 0:
            seq += [torch.cat( [torch.zeros(sz0, shift, device=input.device).long(), input[:,0:sz1-shift]], dim=1 ).unsqueeze(-1)]
         else:
            seq += [input]
      if for_seq:
         return seq
      return torch.cat(seq, dim=2)

   def extra_embed(input, x_emb, params, mode, extra_input):
      assert x_emb is not None
      o = 0
      for i in range(len(x_emb)):
         rsz = x_emb[i]['region']
         x_dsno = x_emb[i]['x_dsno']

         if x_dsno >= 0:
            x_input = gen_bow(extra_input[x_dsno], rsz)
         else:
            x_input = gen_bow(input, rsz)

         xo = embed_plus(x_input, x_emb[i]['params'])

         bias = params.get('embconv.conv%d.b'%i)
         xo = F.conv1d(xo, params['embconv.conv%d.weight'%i], bias=bias, stride=1, padding=0)

         o = o + xo

      return o

   def f(input, params, mode, extra_input=None):
      o = embed(input, params)
      if x_embed is not None:
         o = o + extra_embed(input, x_embed, params, mode, extra_input)

      for i in range(num_block):
         o = block(o, params, 'group.block'+str(i), mode, do_downsample=(i<num_block-1),
                   do_skip_1stAct=(i==0))

      o = o.max(Ldim,keepdim=True)[0]
      o = torch.reshape(o, [o.size()[0], -1])

      if top_dropout != None and top_dropout > 0:
         o = F.dropout(o, top_dropout, mode)

      o = F.linear(o, params['fc.weight'], params['fc.bias'])
      return o

   return f, flat_params

#----------------------------------------------------------
def region_embedding(width, num_classes, vocab_size):
   def embedding_params(vocab_size, out_dim, std=0.01):
      p = {'weight': normal_(torch.Tensor(vocab_size, out_dim), std=std)}
      p['weight'][0,:].fill_(0)
      p['b'] = torch.zeros(1, out_dim)
      return p

   my_params = {
                 'emb': embedding_params(vocab_size, width),
                 'fc': linear_params(width, num_classes)
               }

   flat_params = utils.cast(utils.flatten(my_params))
   flat_params['fc.weight'].unsqueeze_(-1)
   utils.set_requires_grad_except_bn_(flat_params)

   def f(input, params, mode):
      o = embed_plus(input, params)
      o = F.conv1d(o, params['fc.weight'], bias=params['fc.bias'], stride=1, padding=0)
      return o

   return f, flat_params
