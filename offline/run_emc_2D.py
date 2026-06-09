restart = True
refresh_config = False

iters = 100
beta_start = 0.001
beta_stop = 0.2


import pickle
import emc3
from time import time, sleep
import runpy
from pathlib import Path
import h5py
import numpy as np
import sys


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

from emc3.run import calculate_logR, update_models


config_file = 'config.pickle'
config_script = 'config_2D_EMC.py'


if rank == 0 and restart:
    # remove iteration info file
    Path('iteration_info.h5').unlink(missing_ok=True)

    # remove chunked data
    for fnam in Path('./').glob('*chunk*.h5'):
        Path(fnam).unlink(missing_ok=True)

    # remove temporary data files
    for fnam in Path('./').glob('data_*.h5'):
        Path(fnam).unlink(missing_ok=True)

    # remove temporary data files
    for fnam in Path('./').glob('back_*.h5'):
        Path(fnam).unlink(missing_ok=True)

comm.Barrier()

# ensure that only one rank makes the pickle
# file to prevent multiple ranks saving data
if rank == 0 and (restart or refresh_config):
    # make pickle file
    config = runpy.run_path(config_script)['make_pickle_file'](
        output_file='config.pickle')

comm.Barrier()

config = pickle.load(open(config_file, 'rb'))

if restart and rank == 0:
    # initialise model files
    for ci, c in enumerate(config['classes']):
        if c['model'].ndim == 3:
            c['model'].init_blob()
        else:
            c['model'].init_random()

        with h5py.File(c['model_file'], 'w') as f:
            f['data'] = c['model'].data
            f['dq'] = c['model'].dq

    # init fluence
    with h5py.File(config['fluence_file'], 'w') as f:
        D = config['classes'][0]['data'].shape[0]
        f['w_d'] = np.ones(D, dtype=float)

comm.Barrier()

cids = list(range(len(config['classes'])))
my_classes = cids[rank::size]

fnam_iter = Path('iteration_info.h5')

beta = beta_start
last_change = -100
iter_stats = None
Nframes = config['classes'][0]['data'].shape[0]

for i in range(iters):
    # beta scheduling from previous iteration's in-memory stats.
    # Rank 0 holds iter_stats; broadcast just the two change counts.
    n_changes = None
    if rank == 0 and iter_stats is not None:
        n_changes = iter_stats['orientation_changes'] + iter_stats['class_changes']
    n_changes = comm.bcast(n_changes, root=0)

    if n_changes is not None:
        change = n_changes / Nframes
        print(f'{rank=} {change=} {last_change=} {i=}')
        sys.stdout.flush()

        if change < 0.05 and last_change != (i - 1) and beta != beta_stop:
            beta *= 2
            last_change = i

        if beta == beta_stop and change < 0.05 and last_change != (i - 1):
            break

    beta = min(beta, beta_stop)

    t0 = time()
    calculate_logR(config_file, config, p_per_device=2, cids=my_classes)
    time_logR = time() - t0

    comm.Barrier()

    if rank == 0:
        t0 = time()
        stats = emc3.probability.calculate_P(config, beta)
        time_prob = time() - t0

        # save stats to file
        iter_stats = emc3.input_output.save_iteration_info(
            stats, config['working_directory']
        )
        emc3.input_output.print_iteration_stats(iter_stats)

    comm.Barrier()

    # run this, so that we update the fluence for comparison
    t0 = time()
    update_models(config_file, config, p_per_device=2, cids=my_classes)
    time_I = time() - t0

    comm.Barrier()

    if rank == 0:
        emc3.input_output.save_output(config)
        print(f'{time_I=}')
        print(f'{time_prob=}')
        print(f'{time_logR=}')

    comm.Barrier()
