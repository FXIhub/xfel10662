"""
make working directory
    - make sure EMC uses this properly

symlink model file
    - Ery 0.75 nm water?

write config.py

write run_emc.py

sym_link cxi file
"""

from constants import PREFIX, DET_DIST
from pathlib import Path

# working directory
WD = Path(f'{PREFIX}scratch/sizing/r{args.run:04d}/')
WD.mkdir(parents=False, exist_ok=True)

# symlink models
for fnam in {
    'Ery_000nm.h5': 'class_model_0.h5',
    'Ery_050nm.h5': 'class_model_1.h5',
    'Ery_075nm.h5': 'class_model_2.h5',
    'Ery_100nm.h5': 'class_model_3.h5',
    'Ery_150nm.h5': 'class_model_4.h5',
    'Ery_200nm.h5': 'class_model_5.h5',
    'Ery_cluster_contact.h5': 'class_model_6.h5',
    'Ery_pdb.h5': 'class_model_7.h5',
    'Ery_ring_contact.h5': 'class_model_8.h5',
    'Ery_triplet_contact.h5': 'class_model_9.h5',
    'Ery_y_contact.h5': 'class_model_10.h5',
    'Ery_z_contact.h5': 'class_model_11.h5',
    'sphere.h5': 'class_model_12.h5',
