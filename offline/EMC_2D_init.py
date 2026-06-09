import numpy as np
import sys
from pathlib import Path
import shutil

from constants import PREFIX

if __name__ == '__main__':
    cxi_file = sys.argv[1]
    dir_name = sys.argv[2]

    # create directory for this file in scratch
    cxi_dir = Path(PREFIX) / 'scratch' / 'EMC' / dir_name
    cxi_dir.mkdir(parents=True, exist_ok=True)

    # copy cxi file
    shutil.copy(cxi_file, cxi_dir / 'run.cxi')
    shutil.copy('config_2D_EMC.py', cxi_dir)
    shutil.copy('run_emc_2D.py', cxi_dir)
    shutil.copy('submit_2D_EMC.sh', cxi_dir)

    mask_fnam = Path(PREFIX) / 'scratch' / 'det' / 'mask.h5'
    shutil.copy(mask_fnam, cxi_dir)
