"""
Custom Trainer Class
"""
from utils.common import BaseTrainer
import logging
logger_py = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    ''' Trainer object for a3f.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, device=None, vis_dir=None, overwrite_visualization=True, **kwargs):
        pass

    def train_step(self, data, it=None):
        pass

    def eval_step(self, *args, **kwargs):
        pass

    def visualize(self, *args, **kwargs):
        pass