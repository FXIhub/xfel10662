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


# Write per-frame classification back into the CXI as entry_1/result_2
# (CXI v1.6 §A.14 Result: non-image analysis output) with a process_1
# subgroup (§A.13) recording the run. result_2/data is aligned to the CXI
# events; -1 marks events not selected by get_frames() (config_sizing).

# entry_1/
#   result_2/
#       description = '...'
#       data = (D, 3) # N x scale factors for x, y, z
#       frames = (D,) # cxi data indices
#       class_index = (D,) # most likely class index
#       process_1/
#           program = '...'
#           data = '...'
#           version = '...'
#           command = '...'

if rank == 0:
    import datetime
    import sys

    with h5py.File('iteration_info.h5', 'r') as f:
        r_d = f['iteration_0/most_likely_orientation_d'][()]
        c_d = f['iteration_0/most_likely_model_d'][()]

    """
    Internally the mapping matrix will be indexed as:
        M_sr -> M_sjkl:
            s = symmetry index (S_s)
            j = offset index (dr_j)
            k = scale index (scale_k)
            l = orientation index (R_l)

    r -> raveled j,k,l index
    """

    # emc3 indexes data elements 'd' compactly (0..N-1) over the frame subset
    # selected by get_frames(). Recover the d->cxi-event mapping from the
    # loaded DataSparseCXI rather than globbing data_*.h5 (stale files from
    # earlier runs could shadow the current one).

    with h5py.File('run.cxi', 'r+') as f:
        entry = f['entry_1']

        # assume the same for all classes
        frames_d = config['classes'][0]['data'].frames
        D = len(frames_d)
        n_events = entry['instrument_1/detector_1/data'].shape[0]
        c_event = np.full(n_events, -1, dtype=np.int32)
        c_event[frames_d] = c_d.astype(np.int32)

        if 'result_2' in entry:
            del entry['result_2']
        result = entry.create_group('result_2')
        result['description'] = (
            'EMC most-likely scale factors (x,y,z) per CXI event; '
            'class_index = -1 marks events not selected for classification')
        result.create_dataset('class_index', data=c_event)
        result.create_dataset('frames', data=frames_d.astype(np.int64))
        result['detector_1'] = h5py.SoftLink(
            '/entry_1/instrument_1/detector_1')

        # get scale indices per class
        # and scale factors per class
        k_c = []
        scale_c = []
        for c in config['classes']:
            j, k, l = np.indices(c['mapper'].M_sjkl.shape[1:-2])
            k_c.append(k.ravel().copy())
            scale_c.append(c['scale'])

        # get scale index per frame
        k_d = []
        scale_d = []
        for d in range(D):
            cid = c_d[d] # most likely class index
            r = r_d[d]   # most likely r-index

            k = k_c[cid][r] # most likely scale index
            k_d.append(k)

            scale_d.append(scale_c[cid][k])

        # get unique scale factors
        scale_unique = np.unique([c['scale'] for c in config['classes']], axis=0)

        # now map scale_d to the global frame index
        k_event = np.full(n_events, -1, dtype=np.int32)
        k_event[frames_d] = k_d

        scale_event = np.full((n_events, 3), -1, dtype=float)
        scale_event[frames_d] = scale_d

        result['scale_index'] = k_event
        result['data'] = scale_event
        result['scale_unique'] = scale_unique

        process = result.create_group('process_1')
        process['program'] = 'run_emc_sizing.py'
        process['command'] = ' '.join(sys.argv)
        process['date'] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        process['version'] = getattr(emc3, '__version__', 'unknown')
