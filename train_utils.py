import os
import sys
import math

import torch
import torch.nn as nn

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)