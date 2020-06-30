import torch
import numpy as np

from torch.optim import SGD
import torchnet as tnt
import torch.nn.functional as F
from utils.utils import cast
from utils.utils0 import logging, timeLog, Clock, raise_if_nonpositive, raise_if_nan, Global_state, Local_state, stem_name, add_if_absent_, raise_if_absent

#---  constants
Ddim = 0; Cdim = 1
Target_index = 1

#----------------------------------------------------------
# input: opt.ini_type in { iniRand, iniBase, iniBase/2, file, file/2 }.
# output: opt.do_iniBase, opt.fc_scale
#----------------------------------------------------------
def interpret_ini_type_(opt):
   add_if_absent_(opt, ['ini_type','initial'], '')
   opt.do_iniBase = opt.ini_type.startswith('iniBase')
   opt.fc_scale = 0.5 if opt.ini_type.endswith('/2') else -1
   if opt.ini_type.startswith('file') and not opt.initial:
      raise ValueError("ini_type=%s requires 'initial' to specify a pathname of the initial model.")
   if opt.initial and not opt.ini_type.startswith('file'):
      raise ValueError("'initial' requires 'ini_type' to be 'file' or 'file/2'.")
def is_gulf(opt):
   return is_gulf1(opt) or is_gulf2(opt)
def is_gulf1(opt):
   return opt.m > 0
def is_gulf2(opt):
   return (not is_gulf1(opt)) and opt.alpha != 1

#----------------------------------------------------------
def base_update(opt, clock, net, params, trn_data, test_dss, g_st, l_st):
   timeLog('base_update  --------------------------------')

   doing_epo = opt.do_count_epochs
   doing_upd = not doing_epo
   max_count = opt.max_count
   ti_upd,ti_epo = get_test_interval(opt)

   epoch,upd,lr_coeff = l_st.get()
   optimizer = create_optim(opt, lr_coeff, params)

   mtr_loss = tnt.meter.AverageValueMeter()

   #-
   def test_net():
      is_last = (doing_upd and upd >= max_count) or (doing_epo and epoch >= max_count)
      mys = eval(clock, doing_upd, g_st.epo(epoch), g_st.upd(upd), upd, net, test_dss, opt,
                 trn_data=trn_data, params=params, is_last=is_last)
      logging(mys, opt.csv_fn)
   #-
   num_data = 0
   while (doing_upd and upd < max_count) or (doing_epo and epoch < max_count):
      optimizer,lr_coeff = change_lr_if_needed(opt, optimizer, lr_coeff, params, epoch=epoch)

      timeLog('epoch ' + str(epoch) + ' upd ' + str(upd))
      for sample in trn_data:
         num_data += sample[0].size(0)
         if doing_upd and upd >= max_count:
            break
         optimizer,lr_coeff = change_lr_if_needed(opt, optimizer, lr_coeff, params, upd=upd)

         loss,_ = net(sample, is_train=True)
         loss.backward()
         mtr_loss.add(float(loss))

         if opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params.values(), opt.max_grad_norm)

         optimizer.step(); upd += 1; optimizer.zero_grad()

         #---  show progress
         if opt.inc > 0 and upd % opt.inc == 0:
            s = ' ... %d, %.5f, #data,%d' % (upd, mtr_loss.value()[0], num_data)
            timeLog(s)
            raise_if_nan(mtr_loss.value()[0])
            mtr_loss.reset()

         #---  test and save
         is_last = doing_upd and upd >= max_count
         if is_last or ti_upd > 0 and upd % ti_upd == 0:
            test_net()
            save_net(opt, is_last, params, g_st, Local_state(epoch, upd, lr_coeff))

      epoch += 1

      #---  test and save
      is_last = doing_epo and epoch >= max_count
      if is_last or ti_epo > 0 and epoch % ti_epo == 0:
         test_net()
         save_net(opt, is_last, params, g_st, Local_state(epoch, upd, lr_coeff))

   g_st.update(epoch, upd)

#----------------------------------------------------------
def scale_fc(params, name, fc_scale, what=None):
   if what is not None:
      timeLog('Scaling %s of %s by factor %.5f ...' % (name,what,fc_scale))
   else:
      timeLog('Scaling %s by factor %.5f ...' % (name,fc_scale))
   with torch.no_grad():
      for type in ['weight', 'bias']:
         params[name+'.'+type].data *= fc_scale

#----------------------------------------------------------
# base model if num_stages == 1
# base-loop if num_stages > 1
#----------------------------------------------------------
def train_base_model(opt, net, params, trn_data, test_dss):
   check_opt_(opt); show_model_info(opt, params)
   assert opt.alpha == 1 and opt.m <= 0
   if opt.fc_scale > 0:
      raise ValueError('train_base_model: No support fc_scale.')

   clock = Clock()
   g_st = Global_state()
   l_st = Local_state()
   if opt.resume != '':
      g_st,l_st = load(opt.resume, params, do_show_st=True)
   elif opt.initial != '':
      load(opt.initial, params)

   optim = None
   while g_st.lc() < opt.num_stages:
      base_update(opt, clock, net, params, trn_data, test_dss, g_st, l_st)
      l_st.reset()

#----------------------------------------------------------
def train_gulf_model(opt, i_net, i_params, o_net, o_params, trn_data, test_dss):
   check_opt_(opt); show_model_info(opt, o_params)
   raise_if_nonpositive(opt.alpha, 'alpha')
   if is_gulf2(opt):
      if opt.alpha > 1:
         raise ValueError('alpha must be no greater than 1 for GULF2.')

   clock = Clock()
   g_st = Global_state()
   l_st = Local_state()

   if opt.resume != '': # resume training ...
      if opt.do_iniBase or opt.initial:
         logging('!WARNING!: do_iniBase or opt.initial is ignored as resume is given ...')
      logging('Resuming training ... ')
      g_st,l_st = load(opt.resume, o_params, i_params, do_show_st=True)
   else:
      if opt.initial:     # initialize parameter by loading from a file.
         load(opt.initial, o_params, i_params)
      if opt.do_iniBase:  # initialize parameter by regular training.
         base_g_st = Global_state(); base_l_st = Local_state()
         base_update(opt, clock, o_net, o_params, trn_data, test_dss, base_g_st, base_l_st)
         copy_params(src=o_params, dst=i_params)
      if opt.fc_scale > 0: # scale the last linear layer.
         scale_fc(i_params, opt.fc_name, opt.fc_scale, 'i_params')
         scale_fc(o_params, opt.fc_name, opt.fc_scale, 'o_params')

   while g_st.lc() < opt.num_stages:
      gulf_update(opt, clock, i_net, i_params, o_net, o_params,
                  trn_data, test_dss, g_st, l_st)
      copy_params(src=o_params, dst=i_params)
      l_st.reset()

#----------------------------------------------------------
def gulf_update(opt, clock, i_net, i_params, o_net, o_params,
                trn_data, test_dss, g_st, l_st):
   gulf12 = 'GULF1' if is_gulf1(opt) else 'GULF2'
   timeLog('gulf_update (%s) --------------------------------' % gulf12)

   loss_function = o_net(None)

   doing_epo = opt.do_count_epochs
   doing_upd = not doing_epo
   max_count = opt.max_count

   ti_upd,ti_epo = get_test_interval(opt)

   epoch, upd, lr_coeff = l_st.get()
   optimizer = create_optim(opt, lr_coeff, o_params)

   mtr_o_loss = tnt.meter.AverageValueMeter()
   mtr_g_loss = tnt.meter.AverageValueMeter()
   mtr_tar_loss = tnt.meter.AverageValueMeter()

   #-
   def test_o_net():
      is_last = (doing_upd and upd >= max_count) or (doing_epo and epoch >= max_count)
      mys = eval(clock, doing_upd, g_st.epo(epoch), g_st.upd(upd), upd, o_net, test_dss, opt,
                 trn_data=trn_data, params=o_params, is_last=is_last)
      logging(mys, opt.csv_fn)
   #-
   num_data = 0
   while (doing_upd and upd < max_count) or (doing_epo and epoch < max_count):
      optimizer,lr_coeff = change_lr_if_needed(opt, optimizer, lr_coeff, o_params, epoch=epoch)

      logging('epoch ' + str(epoch) + ' upd ' + str(upd))
      for sample in trn_data:
         bsz = sample[0].size(0)
         num_data += bsz
         if doing_upd and upd >= max_count:
            break

         optimizer,lr_coeff = change_lr_if_needed(opt, optimizer, lr_coeff, o_params, upd=upd)

         targets = cast(sample[Target_index], 'long')
         with torch.no_grad():
            i_output = i_net(sample)
         o_loss, o_output = o_net(sample, is_train=True)

         #***  GULF2 (h(p)=L_y(p))
         #     Let f = f_theta and f' = f_{theta_t}
         #     D_{L_y}(f,f') + alpha nabla L_y(f')^T f
         #     = L_y(f) - (1-alpha) nabla L_y(f')^T f + c where c is a constant that does not depend on theta.
         if opt.m <= 0:
            i_output.detach_(); assert i_output.grad is None
            i_output.requires_grad = True
            i_loss = loss_function(i_output, targets)
            i_loss.backward()
            g_loss = o_loss - (1 - opt.alpha)*(o_output*i_output.grad.data).sum()

         #***  GULF1 (h(u)=|u|^2/2)
         else:
            #---  m steps of functional gradient descent
            tar = i_output
            assert tar.size(Ddim) == bsz
            for i in range(opt.m):
               tar.detach_();
               if tar.grad is not None:
                  tar.grad.zero_()
               tar.requires_grad = True
               tar_loss = loss_function(tar, targets)
               tar_loss.backward()
               tar.detach_(); tar.data -= opt.alpha * (bsz*tar.grad.data)

            mtr_tar_loss.add(float(tar_loss))
            #---  || f(theta;x) - f_m^*(x,y) ||^2/2
            g_loss = ((o_output-tar)**2).sum()/2/bsz

         #---
         g_loss.backward()

         mtr_o_loss.add(float(o_loss))
         mtr_g_loss.add(float(g_loss))

         if opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(o_params.values(), opt.max_grad_norm)

         optimizer.step(); upd += 1; optimizer.zero_grad()

         #----  show progress
         if opt.inc > 0 and upd % opt.inc == 0:
            s = '  ... %d, g_loss,%.5f, o_loss,%.5f' % (upd, mtr_g_loss.value()[0], mtr_o_loss.value()[0])
            if mtr_tar_loss.n > 0:
               s += ', t_loss,%.5f' % mtr_tar_loss.value()[0]
            timeLog(s)

            raise_if_nan(mtr_o_loss.value()[0])
            mtr_o_loss.reset(); mtr_g_loss.reset(); mtr_tar_loss.reset()

         is_last = doing_upd and upd >= max_count
         if is_last or ti_upd > 0 and upd % ti_upd == 0:
            test_o_net()
            save_net(opt, is_last, o_params, g_st, Local_state(epoch, upd, lr_coeff),
                     i_params)

      epoch += 1

      is_last = doing_epo and epoch >= max_count
      if is_last or ti_epo > 0 and epoch % ti_epo == 0:
         test_o_net()
         save_net(opt, is_last, o_params, g_st, Local_state(epoch, upd, lr_coeff),
                  i_params)

   g_st.update(epoch, upd)

#----------------------------------------------------------
def save_net(opt, is_end_of_loop, o_params, g_st, l_st, i_params=None):
   fname = opt.save
   if fname == None or fname == '':
      return
   ext = '.pth'
   stem = stem_name(fname, ext)
   save(stem+ext, o_params, g_st, l_st, i_params)
   if opt.do_noow_save and is_end_of_loop:
      slim_save(stem+'-glc'+str(g_st.lc()+1)+'-slim'+ext,
                o_params, g_st, l_st)

#----------------------------------------------------------
def slim_save(fname, o_params, g_st, l_st):
   if fname == None or fname == '':
      return
   timeLog('Saving (slim): ' + fname + ' ... ')
   torch.save(dict(o_params=o_params, i_params=None,
                   global_state=g_st.to_list(), local_state=l_st.to_list(),
                   optimizer=None),
              fname)

#----------------------------------------------------------
def save(fname, o_params, g_st, l_st, i_params=None):
   if fname == None or fname == '':
      return
   timeLog('Saving (fat): ' + fname + ' ... ')
   torch.save(dict(o_params=o_params, i_params=i_params,
                   global_state=g_st.to_list(), local_state=l_st.to_list(),
                   optimizer=None),
              fname)

#----------------------------------------------------------
def load(fname, o_params, i_params=None, do_show_st=False):
   timeLog('Loading ' + fname + ' ... ')

   if torch.cuda.is_available():
      d = torch.load(fname)
   else:
      d = torch.load(fname, map_location='cpu')
   copy_params(src=d['o_params'], dst=o_params)
   if i_params != None:
      if d['i_params'] == None:
         logging('***WARNING***: Copying o_params to i_params since i_params was not saved ...')
         copy_params(src=d['o_params'], dst=i_params)
      else:
         copy_params(src=d['i_params'], dst=i_params)

   g_st = Global_state(inplist=d['global_state'])
   l_st = Local_state(inplist=d['local_state'])

   if do_show_st:
      logging('g_st= %s' % str(g_st))
      logging('l_st= %s' % str(l_st))

   return g_st,l_st

#----------------------------------------------------------
def test(net, data, opt, name):
   timeLog('testing ... '+name)
   topk = [1,5] if opt.do_top5 else [1]
   inc = opt.test_inc
   sum_loss = 0
   mtr_err = tnt.meter.ClassErrorMeter(topk=topk, accuracy=False)

   data_num = 0; count = 0
   for sample in data:
      bsz=sample[0].size(0)
      data_num += bsz; count += 1
      with torch.no_grad():
         loss, output = net(sample, is_train=False)

      mtr_err.add(output.data, sample[Target_index])
      sum_loss += float(loss)*bsz

      if inc > 0 and count % inc == 0:
         s = '... testing ... %d (%d): %s' % (count, data_num, str(float(mtr_err.value()[0])))
         timeLog(s)

   return mtr_err.value(), sum_loss/data_num

#----------------------------------------------------------
def get_loss(net, data, inc, info_max):
   sum_loss = 0; data_num = 0; count = 0
   for sample in data:
      if info_max > 0 and data_num >= info_max:
         break
      bsz=sample[0].size(0)
      data_num += bsz; count += 1
      with torch.no_grad():
         loss, output = net(sample, is_train=False)
      sum_loss += float(loss)*bsz
      if inc > 0 and count % inc == 0:
         timeLog('... getting loss ... %d (%d) ... ' % (count,data_num))

   return sum_loss/data_num

#-----------------------------------------
def get_l2(params):
   sum2 = 0
   for v in params.values():
      if v.requires_grad:
         sum2 += float((v.data**2).sum())
   return sum2

#----------------------------------------------------------
def copy_params(src, dst):
   for key, value in dst.items():
      value.data.copy_(src[key])

#----------------------------------------------------------
def clone_params(src):
   return {
      key: torch.zeros_like(value).data.copy_(value)
      for key, value in src.items()
   }

#----------------------------------------------------------
def print_params(params):
   if len(params) <= 0:
      return
   logging('Parameters: ---------------------------------------------------------------------')
   kmax = max(len(key) for key in params.keys())
   for (key, v) in sorted(params.items()):
      print(key.ljust(kmax+3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)
   logging('---------------------------------------------------------------------------------')

#----------------------------------------------------------
def setup_optim_params(opt, params, lam):
   keys = [ k for k,v in params.items() if v.requires_grad ]
   if lam > 0:
      k_do_reg = keys
      k_dont_reg = []
   else:
      k_do_reg = []
      k_dont_reg = keys

   optim_param =[]
   if len(k_do_reg  ) > 0:
      optim_param += [ {'params': [ v for k,v in params.items() if k in k_do_reg   ], 'weight_decay': lam} ]
   if len(k_dont_reg) > 0:
      optim_param += [ {'params': [ v for k,v in params.items() if k in k_dont_reg ], 'weight_decay': 0.0} ]

   return optim_param,k_do_reg

#----------------------------------------------------------
def create_optim(opt, lr_coeff, params):
   lr = opt.lr*lr_coeff
   lam = opt.weight_decay
   optim_param,_ = setup_optim_params(opt, params, lam)
   timeLog('Creating optimizer with lr=%.5f and lam=%s' % (lr,str(lam)))
   optim = SGD(optim_param, lr, momentum=0.9, weight_decay=lam, nesterov=False)
   optim.zero_grad()
   return optim

#----------------------------------------------------------
def change_lr_if_needed(opt, optimizer, lr_coeff, params, epoch=-1, upd=-1):
   do_change = False
   if upd >= 0 and not opt.do_count_epochs:
      assert epoch == -1
      if upd in opt.decay_lr_at:
         do_change = True
   elif epoch >= 0 and opt.do_count_epochs:
      assert upd == -1
      if epoch in opt.decay_lr_at:
         do_change = True

   if do_change:
      lr_coeff *= opt.lr_decay_ratio
      optimizer = create_optim(opt, lr_coeff, params) # this is from WRN code

   return optimizer, lr_coeff

#----------------------------------------------------------
def show_err_loss(name, pfx, errs, loss):
   s = ","+name+"_err"+pfx
   for err in errs:
      s += ',%.3f' % (err)
   s += ","+name+"_loss"+pfx+","  + '%.5f' % (loss)
   return s

#----------------------------------------------------------
def eval(clock, doing_upd, epo, upd, ite, net, test_dss, opt,
         trn_data, params, is_last, do_mark_last=True):
   clk_tim = clock.suspend()
   pfx = '_t' if is_last and do_mark_last else ''
   s = (',epoch,' + str(epo)) if not doing_upd else ''
   mys = clk_tim + ",upd,"+str(upd) + s + ",ite,"+str(ite)

   for dsinfo in test_dss:
      if opt.do_reduce_testing and dsinfo['name'] == 'test' and not is_last:
         continue

      errs, loss = test(net, dsinfo['data'], opt, dsinfo['name'])
      mys += show_err_loss(dsinfo['name'], pfx, errs, loss)

   if is_last and opt.do_collect_info:
      timeLog('Getting training loss ...')
      trn_loss = get_loss(net, trn_data, opt.test_inc, opt.collect_info_max)
      l2 = get_l2(params)
      mys += ',train_loss%s,%.5f,l2%s,%.3f' % (pfx, trn_loss, pfx, l2)

   clock.resume()
   return mys

#----------------------------------------------------------
def get_test_interval(opt):
   if opt.test_interval <= 0:
      raise ValueError("test_interval must be positive.")

   ti_upd = -1; ti_epo = -1
   if opt.do_count_epochs:
      ti_epo = opt.test_interval
   else:
      ti_upd = opt.test_interval

   return ti_upd, ti_epo

#----------------------------------------------------------
#----------------------------------------------------------
def check_opt_(opt):
   #---  required attributes
   names = [ 'do_iniBase','fc_scale','alpha','m','num_stages','weight_decay','max_count','do_count_epochs','lr','test_interval']
   raise_if_absent(opt, names, who='gulf')   

   add_if_absent_(opt, ['fc_name'], 'fc')

   #---  optional attributes
   add_if_absent_(opt, ['do_reduce_testing','do_top5','do_collect_info','do_noow_save','verbse'], False)
   add_if_absent_(opt, ['inc','test_inc','collect_info_max'], -1)
   add_if_absent_(opt, ['max_grad_norm','lr_decay_ratio'], -1)
   add_if_absent_(opt, ['resume','save','initial','csv_fn'], '')
   add_if_absent_(opt, ['decay_lr_at'],[])

   if opt.verbose:
      logging('gulf.check_opt_: opt -------------------------------------------')
      logging({**vars(opt)})
      logging('----------------------------------------------------------------')

#----------------------------------------------------------
def show_model_info(opt, params):
   if opt.verbose:
      print_params(params)
   n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
   logging('#parameters:' + str(n_parameters))