import numpy as np
import torch
import os
import random
import logging

def set_device(device_num):
    if device_num == -1 or torch.cuda.is_available() == False:
        device = torch.device('cpu')
    elif torch.cuda.device_count() == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:{}'.format(int(device_num)))
    return device

# random seed controller
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = torch.cuda.is_available()

# default logger setter
def set_logger(config):
    logfile = os.path.join(config.out_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)


# Base Trainer class
class BaseTrainer(object):
    """ Base trainer class.
    """
    def evaluate(self, *args, **kwargs):
        """ Performs an evaluation.
        """
        raise NotImplementedError

    def train_step(self, *args, **kwargs):
        """ Performs a training step.
        """
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        """ Performs an evaluation step.
        """
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        """ Performs  visualization.
        """
        raise NotImplementedError
