from pathlib import Path
import numpy as np
import h5py

working_directory = Path(__file__).parent

cxi_file = 'run.cxi'


def get_frames(hit_sigma_threshold=10.0, max_frames=1024):
    """
    """
    sig = None
    with h5py.File(cxi_file) as f:
        D = f['entry_1/data_1/data'].shape[0]

        k0 = 'entry_1/instrument_1/detector_1/hit_sigma'
        k1 = 'entry_1/instrument_1/detector_1/score/hit_sigma'
        if k0 in f:
            sig = f[k0][()]

        elif k1 in f:
            sig = f[k1][()]

        else:
            print(f'warning {k0} and {k1} not found in {cxi_file}, skipping sigma threshold')


    if sig is not None:
        i = np.argsort(sig)[::-1][:max_frames]
        i = i[sig[i] > hit_sigma_threshold]
        frames = i
    else:
        frames = np.arange(D)

    print(f'selected frames: {frames}')
    return frames


def make_pickle_file(
        shape=(192, 192, 192),
        rotation_order=6,
        frames=None,
        P_thresh=0.,
        classes=1,
        classes_2D=0,
        pixels_per_voxel=2,
        P_mask_padding = (1, 32),
        likelihood='fluence_free',
        #likelihood='Poisson',
        update_model=False,
        output_file='config.pickle'):
    """
    make 6 face-view symmetry classes (D6) in a continuity group
    make 6 x side-view symmetry classes (inv.) in a continuity group
    make 6 y side-view symmetry classes (inv.) in a continuity group
    make 6 y symmetry classes (inv.) in a continuity group
    """

    import numpy as np
    import pickle
    import emc3
    import h5py

    if frames is None:
        frames = get_frames()

    fluence = np.ones(len(frames), dtype=float)

    with h5py.File('mask.h5') as f:
        mask = f['data'][()]

    det = emc3.Detector_cxi(cxi_file, mask)

    # make a 3D model
    # ---------------
    # here the dq of the model depends
    # on the detector pixel size
    x0 = 400e-6
    y0 = 400e-6
    dx = 400e-6

    xyz_offset = [[dx, dx, 0]]

    scale = [1.0]

    dq = det.dq * pixels_per_voxel
    model = emc3.Model(dq=dq, shape=shape, symmetry='D6')

    # make pixels masks that account for:
    # model size, padding, scaling and offsets

    # make a Probability calc mask
    P_mask = emc3.make_mask(
            det,
            model,
            xyz_offset,
            [1,],
            P_mask_padding)

    # data object for probability calc
    P_K_di = emc3.DataSparseCXI(cxi_file, det, P_mask, frames, background=True)

    # included loaded data in pickle file
    P_K_di.load_data()
    P_K_di.save_to_file()
    P_K_di.unload()  # free memory for saving

    # make a mask for model updates
    mask = emc3.make_mask(
            det,
            model,
            xyz_offset,
            [1,])

    # data object for model update
    K_di = emc3.DataSparseCXI(cxi_file, det, mask, frames, background=True)
    K_di.load_data()
    K_di.save_to_file()
    K_di.unload()  # free memory for saving

    # make a mapper for detector pixel --> model voxels
    # and vice versa
    # scale factors for (x,y) and z separately
    # +- 40%
    scale = []
    sizes_ss = np.linspace(0.6, 1.4, 17)
    sizes_fs = np.linspace(0.6, 1.4, 17)
    for sx in sizes_ss:
        for sz in sizes_fs:
            scale.append([sx, sx, sz])

    mapper= emc3.Mapper(
            det,
            model,
            rotation_order,
            scale=scale,
            offsets=xyz_offset)

    config = {}
    config['working_directory'] = working_directory
    config['classes'] = []
    config['continuity_groups'] = []

    for c in range(classes):
        symmetry = 'D6'
        mapper_c = mapper
        model_c = emc3.Model(dq=dq, shape=shape, symmetry=symmetry, class_id=c)

        config['classes'].append(
                {
                'class_id': c,
                'cxi_file': cxi_file,
                'P_data': P_K_di,
                'data': K_di,
                'model': model_c,
                'fluence': fluence,
                'mapper': mapper_c,
                'interpolation_forward': 'linear',
                'likelihood': likelihood,
                'frame_model': 'background',
                'maximise': 'I',
                'update_model': update_model,
                'filter_model': None,
                'P_thresh': P_thresh,
                'scale': scale,
                'scale_fs': sizes_fs,
                'scale_ss': sizes_ss,
                }
        )

    # update default parameters for each class
    config = emc3.make_config(config)

    pickle.dump(config, open(output_file, 'wb'))

    return config

if __name__ == '__main__':
    make_pickle_file()


