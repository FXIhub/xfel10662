import pathlib
import os

from constants import EXP_ID

# .../xfel7927/
root = pathlib.Path(__file__).parent.parent.resolve()

"""
from extra_data import open_run
run = open_run(proposal=10662, run=7)

motor_z = run[det_pos, 'encoderPosition'].drop_empty_trains()[0].ndarray()[0]

# actual sample-detector distance in metres
actual_z = DET_OFFSET + 1e-3 * motor_z
"""
DET_OFFSET=0.4995748960790368

def get_geom(run_no):
    return make_geom(run_no)

def make_geom(run_no):
    out_fnam = root.joinpath(f'geom/r{run_no:04}.geom').resolve()
    ref_fnam = root.joinpath('geom/agipd_p10662_pw_70cm_from_260505.geom').resolve()

    if not os.path.exists(out_fnam) :
        from extra_geom import AGIPD_1MGeometry
        from extra_geom.motors import AGIPD_1MMotors
        from extra.components import AGIPD1MQuadrantMotors
        from extra_data import open_run

        run = open_run(int(EXP_ID), run_no)

        det_pos = "SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER"
        motor_z = run[det_pos, 'encoderPosition'].drop_empty_trains()[0].ndarray()[0]
        actual_z = DET_OFFSET + 1e-3 * motor_z

        ref_geom = AGIPD_1MGeometry.from_crystfel_geom(ref_fnam)
        motors = AGIPD1MQuadrantMotors(run)
        tracker = AGIPD_1MMotors(ref_geom)
        geom = tracker.geom_at(motors.most_frequent_positions())

        z_mean = np.mean(geom.get_pixel_positions()[..., 2])

        geom = geom.offset((0, 0, actual_z - z_mean))

        geom.write_crystfel_geom(out_fnam)

    return out_fnam
