"""
   Original: https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
   Restructured and dropout was added.
"""
import torch
import torch.nn.functional as F
import utils.utils as utils

def gen_block_params(ni, no):
   return {
      'conv0': utils.conv_params(ni, no, 3),
      'conv1': utils.conv_params(no, no, 3),
      'bn0': utils.bnparams(ni),
      'bn1': utils.bnparams(no),
      'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
   }

def gen_group_params(ni, no, count):
   return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
           for i in range(count)}

def group(o, block_f, params, base, mode, stride, gdep, dropout):
   for i in range(gdep):
      o = block_f(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1, dropout)
   return o

def block(x, params, base, mode, stride, dropout):
   o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
   o = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
   o = F.relu(utils.batch_norm(o, params, base + '.bn1', mode), inplace=True)
   if dropout != None and dropout > 0:
      o = F.dropout(o, dropout, mode)
   o = F.conv2d(o, params[base + '.conv1'], stride=1, padding=1)

   if base + '.convdim' in params:
      return o + F.conv2d(o1, params[base + '.convdim'], stride=stride)
   else:
      return o + x

#----------------------------------------------------------------------
def resnet(dep, kk, num_classes, dropout=-1):
   if dep <= 4:
      raise ValueError('depth must be group_depth*6+4.')

   if (dep-4)%6 != 0:
      raise ValueError('depth should be 6n+4')

   gdep = (dep - 4) // 6
   return resnet3g(gdep, num_classes, imgsz=32, wid0=16, wid1=int(16*kk), p0=3, dropout=dropout)

#----------------------------------------------------------------------
def resnet3g(gdep, num_classes, imgsz=32, wid0=16, wid1=16, p0=3, dropout=-1):

   widths = [wid1, wid1*2, wid1*4]

   flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, wid0, p0),
        'group0': gen_group_params(wid0, widths[0], gdep),
        'group1': gen_group_params(widths[0], widths[1], gdep),
        'group2': gen_group_params(widths[1], widths[2], gdep),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
   }))

   utils.set_requires_grad_except_bn_(flat_params)

   def f(input, params, mode):
      o = F.conv2d(input, params['conv0'], padding=1)
      g0 = group(o, block, params, 'group0', mode, 1, gdep, dropout)
      g1 = group(g0, block, params, 'group1', mode, 2, gdep, dropout)
      g2 = group(g1, block, params, 'group2', mode, 2, gdep, dropout)
      g2r = F.relu(utils.batch_norm(g2, params, 'bn', mode))
      o = F.avg_pool2d(g2r, 8, 1, 0)
      o = o.view(o.size(0), -1)
      return F.linear(o, params['fc.weight'], params['fc.bias'])

   return f, flat_params
