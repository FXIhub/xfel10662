from pathlib import Path
import numpy as np
import h5py

working_directory = Path(__file__).parent

cxi_file = 'run.cxi'


def get_frames(hit_sigma_threshold=200.0, max_frames=100000):
    """
    """
    sig = None
    with h5py.File(cxi_file) as f:
        k = 'entry_1/instrument_1/detector_1/score/hit_score_mask'
        if k in f:
            sig = f[k][()]

        else:
            print(f'warning {k} not found in {cxi_file}, skipping sigma threshold')


    i = np.argsort(sig)[::-1][:max_frames]
    i = i[sig[i] > hit_sigma_threshold]
    frames = i

    print(f'selected frames: {frames}')
    return frames


def make_pickle_file(
        shape=(192, 192),
        rotation_order=10,
        frames=None,
        P_thresh=0.,
        classes=0,
        classes_2D=32,
        pixels_per_voxel=2,
        P_mask_padding = (1, 8),
        likelihood='fluence_free', # Poisson, fluence_free
        #likelihood='Poisson',
        frame_model='background', # basic, background
        update_model=True,
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
    xyz_offset = [[0, 0, 0]]

    scale = [1.0]

    dq = det.dq * pixels_per_voxel
    model_inv = emc3.Model(dq=dq, shape=shape, symmetry='inversion')

    # make pixels masks that account for:
    # model size, padding, scaling and offsets

    # make a Probability calc mask
    P_mask = emc3.make_mask(
            det,
            model_inv,
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
            model_inv,
            xyz_offset,
            [1,])

    # data object for model update
    K_di = emc3.DataSparseCXI(cxi_file, det, mask, frames, background=True)
    K_di.load_data()
    K_di.save_to_file()
    K_di.unload()  # free memory for saving

    # make a mapper for detector pixel --> model voxels
    # and vice versa
    #scale_offset = -0.1  # -10%
    #scale_offset = -0.0  # -10%
    #scale = scale_offset + np.array([0.98, 0.97, 0.98, 0.99, 1.00])
    scale = [1.]
    mapper_inv = emc3.Mapper(
            det,
            model_inv,
            rotation_order,
            scale=scale,
            offsets=xyz_offset)


    config = {}
    config['working_directory'] = working_directory
    config['classes'] = []
    config['continuity_groups'] = []

    for c in range(classes_2D):
        symmetry = 'inversion'
        mapper_c = mapper_inv

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
                'frame_model': frame_model,
                'maximise': 'I',
                'update_model': update_model,
                'filter_model': None,
                'P_thresh': P_thresh,
                }
        )

    # update default parameters for each class
    config = emc3.make_config(config)

    pickle.dump(config, open(output_file, 'wb'))

    return config

if __name__ == '__main__':
    make_pickle_file()


