import pickle
import emc3
from time import time, sleep
import runpy
from pathlib import Path
import h5py
import numpy as np


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

from emc3.run import calculate_logR, update_models

restart = True
refresh_config = False

config_file = 'config.pickle'
config_script = 'config.py'
iters = 1
beta = 0.1


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
    # init fluence
    with h5py.File(config['fluence_file'], 'w') as f:
        D = config['classes'][0]['data'].shape[0]
        f['w_d'] = np.ones(D, dtype=float)

comm.Barrier()

cids = list(range(len(config['classes'])))
my_classes = cids[rank::size]

fnam_iter = Path('iteration_info.h5')

for i in range(iters):
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
