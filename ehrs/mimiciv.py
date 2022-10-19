import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ehrs import register_ehr, EHR
from utils import utils

logger = logging.getLogger(__name__)

@register_ehr('mimiciv')
class MIMICIV(EHR):
    def __init__(self, data, cfg):
        super().__init__()
        self.cfg = cfg

        self.dir_path = data
        