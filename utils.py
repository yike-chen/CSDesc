import os,sys
import numpy as np
from copy import deepcopy
import torch
# from tqdm import tqdm
import logging
import time
from pathlib import Path

########################################################################################################################
def create_logger(file_path, opt):

    logger_obj = logging.getLogger()
    logger_obj.setLevel(logging.INFO)

    root_output_dir = Path(file_path)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    if opt.contilearn:
        log_file = '{}_{}_{}_{}_{}.log'.format(time_str, opt.arch, opt.pooling, opt.mode, 'owm')
    else:
        log_file = '{}_{}_{}_{}.log'.format(time_str, opt.arch, opt.pooling, opt.mode)
    final_log_file = root_output_dir / log_file

    fh = logging.FileHandler(str(final_log_file))
    ch = logging.StreamHandler()
    
    formater = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formater)
    ch.setFormatter(formater)

    logger_obj.addHandler(fh)
    logger_obj.addHandler(ch)

    return logger_obj


########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
