"""DAMNIT context file for SPB/p010662 (xfel10662).

First-pass smoke test: report first/last train timestamps and number of
trains per run. DAMNIT reserves `start_time` as a built-in column
(populated from run migration metadata as a Unix timestamp), so we use
distinct names here to avoid colliding with it.

Developed in this repo for version control; deploy by copying (or
symlinking) into a DAMNIT directory on Maxwell.
"""
from datetime import timedelta
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import xarray as xr

from damnit_ctx import Variable, Skip


# DAMNIT exec()s this file without setting __file__, so recover the real path
# from the running code object. .resolve() then follows the sandbox symlink to
# the actual repo location, and we walk up to the offline/ scripts dir.
try:
    _here = Path(__file__).resolve()
except NameError:
    import sys as _sys
    _here = Path(_sys._getframe(0).f_code.co_filename).resolve()
OFFLINE_DIR = _here.parent.parent / "offline"

# Undulator
undulator = "SPB_XTD2_UND/DOOCS/ENERGY"
# XTD2 attenuator
xtd2_att = "SA1_XTD2_ATT/MDL/MAIN"
# XTD9 attenuator
xtd9_att = "SPB_XTD9_ATT/MDL/MAIN"
# XTD2 XGM
xtd2_xgm = "SA1_XTD2_XGM/XGM/DOOCS"
# XTD9 XGM
xtd9_xgm = "SPB_XTD9_XGM/XGM/DOOCS"
# Bunch pattern decoder
bunch_pattern = "SPB_RR_SYS/MDL/BUNCH_PATTERN"
# Detector Z motor
det_pos = "SPB_IRU_AGIPD1M/MOTOR/Z_STEPPER"
# Detector control
det_ctrl = "SPB_IRU_AGIPD1M1/MDL/FPGA_COMP"

# Inline camera
inlinemic_cam = "SPB_IRU_INLINEMIC/CAM/CAM"
# Sidemic camera
sidemic_cam = "SPB_EXP_ZYLA/CAM/1"
# Taylor cone camera
taylor_cam = "SPB_IRU_AEROSOL/CAM/CAM_1"

# HPLC pumps
hplc_pump = "SPB_BIO_SYS/PUMP/HPLC"
sample_pump = "c"
# Flowmeters
jet_flowmeter = "SPB_IRD_LIQUIDJET/FLOW/USER_He"
cube_flowmeter = "SPB_IRD_LIQUIDJET/FLOW/CUBE_He"

n2_cap_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_N2_CAPILLARY"
co2_cap_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_CO2_CAPILLARY"

n2_chm_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_N2_CHAMBER"
co2_chm_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_CO2"
he_chm_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_HE_CHAMBER"

dp_inner_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_DP"
dp_outer_flowmeter = "SPB_IRU_LIQUIDJET/FLOW/ESI_DP_2"

esi_hv = "SPB_EXP_HV/MDL/SHQ1"

# Electrospray / aerosol injector sources per Johan Bielecki (SPB/SFX) email,
# 2026-04-17, "Accessing electrospray parameters from recorded data". These are
# the SPB_IRU_AEROSOL/FLOW devices actually archived in the beamtime data, which
# differ from the SPB_IRU_LIQUIDJET/FLOW/ESI_* names used in the commented blocks
# below. CO2_capillary, N2_capillary and He_chamber were unused in that beamtime
# (constant/zero) but are included for completeness. All are per-train scalars.
aerosol_co2_cap = "SPB_IRU_AEROSOL/FLOW/CO2_CAPILLARY"
aerosol_co2_chm = "SPB_IRU_AEROSOL/FLOW/CO2_CHAMBER"
aerosol_dp1     = "SPB_IRU_AEROSOL/FLOW/DP_1"
aerosol_dp2     = "SPB_IRU_AEROSOL/FLOW/DP_2"
aerosol_he_chm  = "SPB_IRU_AEROSOL/FLOW/HE_CHAMBER"
aerosol_n2_cap  = "SPB_IRU_AEROSOL/FLOW/N2_CAPILLARY"
aerosol_n2_chm  = "SPB_IRU_AEROSOL/FLOW/N2_CHAMBER"
liquidjet_he    = "SPB_IRU_LIQUIDJET/FLOW/HE"
# Taylor cone / aerosol camera (taylor_cam above): SPB_IRU_AEROSOL/CAM/CAM_1


HITFINDER_SOURCE = 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output'

# The facility's SPI hit-finder writes HITFINDER_SOURCE into the proc data some
# time *after* the corrected detector data appears, so a run can have proc data
# while the hit-finder source is still missing. The CXI step and everything
# downstream read this source, so we gate them on a variable that blocks until
# it shows up (see spi_hitfinder_ready). Tunables:
HITFINDER_POLL_INTERVAL_S = 60        # how often to re-check proc
HITFINDER_TIMEOUT_S = 4 * 60 * 60     # give up after this long (slurm_time is 5h)


def proc_has_hitfinder(proposal_path, run_no):
    """Re-open the proc run from disk and report whether the SPI hit-finder
    source is present with at least one frame. Re-opening on every call (rather
    than reusing the DAMNIT `run` object, which is a snapshot from when the
    extraction started) is required so newly written proc files are picked up."""
    from extra_data import RunDirectory
    proc_dir = proposal_path / "proc" / f"r{run_no:04d}"
    if not proc_dir.is_dir():
        return False
    try:
        r = RunDirectory(proc_dir)
    except Exception:
        return False
    if HITFINDER_SOURCE not in r.all_sources:
        return False
    try:
        return r[HITFINDER_SOURCE, 'data.hitFlag'].shape[0] > 0
    except Exception:
        return False


# search log files
def search_file(string, fnam):
    search_word = string.lower()

    with open(fnam, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            if search_word in line.lower():
                return True
    return False

def run_slurm_script_as_bash(script, *args):
    try:
        # Runs the script, blocks until finished, and captures output/errors
        result = subprocess.run(
            ["bash", script] + [str(arg) for arg in args],
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        raise Skip(f'{script} failed (rc={e.returncode}): {e.stderr or e.stdout}')

    return 'Done'

def run_and_check_logs(name, run_no, proposal_path):
    # check log
    #SBATCH -o ${EXP_PREFIX}/scratch/log/vds-%a.out
    vds_log = proposal_path / "scratch" / "log" / f"{name}-{run_no}.out"
    vds_slurm_script = OFFLINE_DIR / f'submit_{name}.sh'

    if vds_log.is_file():
        is_done = search_file(f'{name} done', vds_log)

        # nothing to do
        if is_done:
            return 'Done'

        # there is an error and we shouldn't keep running
        else:
            raise Skip(f"Error check {vds_log}")

    # run vds sbatch job blocking
    else:
        return run_slurm_script_as_bash(vds_slurm_script, run_no)

# Run metadata

@Variable(title="Trains")
def n_trains(run):
    return len(run.train_ids)

@Variable(title="Run length")
def run_length(run):
    # Identify the list of 'valid' train ids
    tid_sets = [f.valid_train_ids for f in run.files]
    valid_train_ids = sorted(set().union(*tid_sets))

    ts = run.train_timestamps()
    delta = ts[run.train_ids == valid_train_ids[-1]] - ts[run.train_ids == valid_train_ids[0]]
    delta_s = int(delta / np.timedelta64(1, 's'))
    return str(timedelta(seconds=delta_s))

def run_size_tb(path):
    run_size_bytes = sum(f.stat().st_size for f in path.rglob('*'))
    return run_size_bytes / 1e12

@Variable(title="Raw size (TB)")
def raw_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "raw" / f"r{run_no:04}"
    return run_size_tb(run_path)

@Variable(title="Proc size (TB)", data="proc")
def proc_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "proc" / f"r{run_no:04}"
    return run_size_tb(run_path)

# Beam properties

@Variable(title="Photon energy (keV)", summary="mean")
def photon_energy(run):
    return run[undulator, 'actualPosition'].drop_empty_trains()[0].ndarray()

@Variable(title="Pulses", summary="mean")
def n_pulses(run):
    return run[bunch_pattern, 'sase1.nPulses'].ndarray()

@Variable(title="Rep. rate (MHz)")
def repetition_rate(run):
    # sase1.nPulses varies over the run (0-pulse trains while the beam is off,
    # full pattern while on), so as_single_value() can't collapse it to one
    # value. Use the maximum, i.e. the rate while the beam is actually
    # delivering, and read pulse IDs from that full-pattern train.
    n_pulses_arr = run[bunch_pattern, 'sase1.nPulses'].ndarray()
    n_pulses = int(n_pulses_arr.max())

    if n_pulses == 0:
        return 0.0
    if n_pulses == 1:
        return 1.0e-6

    full_train = int(np.argmax(n_pulses_arr))
    pulse_ids = run[bunch_pattern, 'sase1.pulseIds'].ndarray()[full_train, :n_pulses]

    # compute repetition rate (in MHz) based on pulse ID spacing
    pulseIdsIncr = np.gradient(pulse_ids)
    if np.all(pulseIdsIncr == pulseIdsIncr[0]):
        return 1300. / 288. / pulseIdsIncr[0]
    return np.nan

@Variable(title="XGM intensity (uJ)", summary="mean")
def xgm_intensity(run):
    n_pulses = run[xtd2_xgm, 'pulseEnergy.numberOfSa1BunchesActual'].drop_empty_trains().ndarray()[0].squeeze()
    pulse_energy = run[f"{xtd2_xgm}:output", 'data.intensitySa1TD'].drop_empty_trains().ndarray()[..., :n_pulses].mean(axis=1)
    return pulse_energy

@Variable(title="Transmission")
def transmission(run):
    xtd2_t = run[xtd2_att, 'actual.transmission'].drop_empty_trains()[0].ndarray()
    xtd9_t = run[xtd9_att, 'actual.transmission'].drop_empty_trains()[0].ndarray()
    total_t = (xtd2_t * xtd9_t).squeeze()
    return total_t

# Detector

@Variable(title="Detector pos. (mm)")
def det_position(run):
    motor_z = run[det_pos, 'encoderPosition'].drop_empty_trains()[0].ndarray()[0]
    DET_OFFSET = 0.4995748960790368  # metres; offset between motor zero and true sample-detector distance
    # motor_z (encoderPosition) is in mm; report the true sample-detector distance in mm
    return DET_OFFSET * 1e3 + motor_z

@Variable(title="AGIPD memory cells")
def agipd_memory_cells(run):
    return run[det_ctrl, 'bunchStructure.nPulses'].drop_empty_trains()[0].ndarray()[0]

@Variable(title="AGIPD rep. rate (MHz)")
def agipd_rep_rate(run):
    return run[det_ctrl, 'bunchStructure.repetitionRate'].drop_empty_trains()[0].ndarray()[0]

@Variable(title="AGIPD int. time")
def agipd_int_time(run):
    return run[det_ctrl, 'integrationTime'].drop_empty_trains()[0].ndarray()[0]

@Variable(title="AGIPD gain mode")
def agipd_gain_mode(run):
    return run[det_ctrl, 'gainModeIndex'].drop_empty_trains()[0].ndarray()[0]

# Sample delivery

##@Variable(title="Inline image")
##def inline_image(run):
##    return run[f"{inlinemic_cam}:daqOutput", 'data.image.pixels'].ndarray()[0]

##@Variable(title="Side image")
##def side_image(run):
##    return run[f"{sidemic_cam}:daqOutput", 'data.image.pixels'].ndarray()[0]

@Variable(title="Taylor cone image")
def taylor_cone_image(run):
    return run[f"{taylor_cam}:daqOutput", 'data.image.pixels'].ndarray()[0]

# Electro spray (sources per Johan Bielecki email, 2026-04-17)

@Variable(title="N2 flow in chamber", summary="mean")
def n2_flow_chamber(run):
    return run[aerosol_n2_chm, "measureCapacity.value"].ndarray()

@Variable(title="CO2 flow in chamber", summary="mean")
def co2_flow_chamber(run):
    return run[aerosol_co2_chm, "measureCapacity.value"].ndarray()

@Variable(title="He flow in chamber", summary="mean")
def he_flow_chamber(run):
    return run[aerosol_he_chm, "measureCapacity.value"].ndarray()

@Variable(title="N2 flow in capillary", summary="mean")
def n2_flow_capillary(run):
    return run[aerosol_n2_cap, "measureCapacity.value"].ndarray()

@Variable(title="CO2 flow in capillary", summary="mean")
def co2_flow_capillary(run):
    return run[aerosol_co2_cap, "measureCapacity.value"].ndarray()

@Variable(title="dP: outer capillary (DP_1)", summary="mean")
def dp_outer(run):
    return run[aerosol_dp1, "measureCapacity.value"].ndarray()

@Variable(title="Liquidjet He flow", summary="median")
def liquidjet_he_flow(run):
    return run[liquidjet_he, "measureCapacity.value"].ndarray()

@Variable(title="Electrospray current (A)", summary="mean")
def esi_current(run):
    return run[esi_hv, "channel1.current.value"].ndarray()

@Variable(title="Electrospray voltage (V)", summary="mean")
def esi_voltage(run):
    return run[esi_hv, "channel1.voltage.value"].ndarray()

# @Variable(title="Sample set flow (μL/min)", summary="mean")
# def sample_flow(run):
#     return run[hplc_pump, f'{sample_pump}.targetFlow.value'].ndarray() * 1.e3

# @Variable(title="Sample pressure (psi)", summary="mean")
# def sample_pressure(run):
#     return run[hplc_pump, f'{sample_pump}.pressure.value'].ndarray()

# @Variable(title="Jet gas flow (mg/min)", summary="mean")
# def jet_gas_flow(run):
#     return run[jet_flowmeter, 'measureCapacity.value'].ndarray()

# @Variable(title="Cube gas flow (L/min)", summary="mean")
# def cube_gas_flow(run):
#     return run[cube_flowmeter, 'measureCapacity.value'].ndarray()


### Values from myMDC

@Variable(title="Run type")
def get_run_type(run, run_type: "mymdc#run_type"):
    return run_type

@Variable(title="Sample Name")
def get_run_sample(run, run_sample: "mymdc#sample_name"):
    return run_sample


### SPI Hit Finder output

@Variable(title="Hit rate, %", data="proc")
def spi_hit_rate(run, gate: "var#spi_hitfinder_ready"):
    hit_arr = run[HITFINDER_SOURCE, 'data.hitFlag'].ndarray()
    n_frames = hit_arr.size
    n_hits = np.sum(hit_arr)
    return (n_hits/n_frames) * 100

@Variable(title="Num. hits", data="proc", summary="sum")
def num_hits(run, gate: "var#spi_hitfinder_ready"):
    hit_arr = run[HITFINDER_SOURCE, 'data.hitFlag'].ndarray()
    tr_id_arr = run[HITFINDER_SOURCE, 'data.trainId'].ndarray()

    tr_ids, ix = np.unique(tr_id_arr, return_inverse=True)
    hits_per_train_arr = np.bincount(ix, weights=hit_arr, minlength=len(tr_ids))

    hits_per_train_xarr = xr.DataArray(
        hits_per_train_arr, dims=['trainId'], coords={'trainId': tr_ids}
    )

    return hits_per_train_xarr

@Variable(title="Hitscore", data="proc", summary="mean")
def spi_hitscore(run, gate: "var#spi_hitfinder_ready"):
    hitscore = run[HITFINDER_SOURCE, 'data.hitscore'].ndarray()
    tid = run[HITFINDER_SOURCE, 'data.trainId'].ndarray()
    hitscore_xarr = xr.DataArray(
        hitscore, dims=['trainId'], coords={'trainId': tid}
    )
    return hitscore_xarr

@Variable(title="SPI hitfinder ready", data="proc")
def spi_hitfinder_ready(run, run_no: "meta#run_number",
                        proposal_path: "meta#proposal_path"):
    """Gate the whole offline pipeline on the facility's hit-finder output.

    The hit-finder (REDU) source is written into proc only after the corrected
    detector data is complete, so its presence is our signal that proc is ready.
    Block here (polling the proc run on disk) until it appears, then let VDS run
    -- this prevents VDS from building a short/partial virtual dataset off
    half-written CORR files. Raise Skip on timeout so DAMNIT doesn't mark the run
    done with a half-built pipeline; a later reprocess will retry.
    """
    deadline = time.monotonic() + HITFINDER_TIMEOUT_S
    while not proc_has_hitfinder(proposal_path, run_no):
        if time.monotonic() >= deadline:
            raise Skip(
                f"SPI hit-finder source {HITFINDER_SOURCE} still absent from "
                f"proc for r{run_no} after {HITFINDER_TIMEOUT_S / 3600:.1f} h")
        time.sleep(HITFINDER_POLL_INTERVAL_S)
    return 'Done'

@Variable(title="VDS", data="proc")
def make_vds(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
        gate: "var#spi_hitfinder_ready"):
    """Run VDS if the logfile is not present. Gated on spi_hitfinder_ready so it
    only builds once the corrected proc data is complete."""
    return run_and_check_logs('vds', run_no, proposal_path)

@Variable(title="CXI", data="proc")
def make_cxi(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
        vds_state: "var#make_vds"):
    return run_and_check_logs('cxi', run_no, proposal_path)

@Variable(title="Background radial profile", data="proc", summary="median")
def background_radial_profile(run, run_no: "meta#run_number",
                             proposal_path: "meta#proposal_path",
                             cxi_state: "var#make_cxi"):
    """Azimuthal average of the beamline background (data_white) over good
    pixels, written into the CXI file and stored here as a 1D array vs radius.
    The table summary is its median value. Shares one implementation with the
    CLI script offline/add_background_radial_profile.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "add_background_radial_profile",
        OFFLINE_DIR / "add_background_radial_profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cxi_file = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi"

    r, B_r = mod.add_background_radial_profile(cxi_file)
    return xr.DataArray(B_r, dims=['radius'], coords={'radius': r})

@Variable(title="Classification", data="proc")
def make_classification(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
        cxi_state: "var#make_cxi"):
    if run_and_check_logs('classification', run_no, proposal_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_histogram", OFFLINE_DIR / "make_histogram.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        info = (proposal_path / "scratch" / "classification"
                / f"r{run_no:04d}" / "iteration_info.h5")
        fig = mod.make_histogram_figure(info)
        fig.savefig(info.parent / "classification_histogram.png")
    return fig

@Variable(title="Sizing", data="proc")
def make_sizing(run, run_no: "meta#run_number",
                             proposal_path: "meta#proposal_path",
                             class_state: "var#make_cxi"):
    """2D histogram of most-likely class assignments. Loaded from
    offline/make_sizing_histogram.py so the CLI script and the DAMNIT cell share one
    implementation. Also drops a PNG next to iteration_info.h5 for offline
    viewing/transfer."""
    if run_and_check_logs('sizing', run_no, proposal_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_sizing_histogram", OFFLINE_DIR / "make_sizing_histogram.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cxi_file = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi"

        fig = mod.make_sizing_histogram(cxi_file, run_no)

        # write figure to sizing folder
        directory = proposal_path / "scratch" / "sizing" / f"r{run_no:04d}"
        fig.savefig(directory / "sizing_histogram.png")
    return fig

@Variable(title="Good Hits", summary="sum")
def good_hits(run, run_no: "meta#run_number",
                             proposal_path: "meta#proposal_path",
                             class_state: "var#make_classification",
                             size_state: "var#make_sizing"):
    """Filter hits by classification and size fit and Hitscore."""
    import h5py
    hit_score_threshold = 500.
    size_min = 0.7
    size_max = 1.2
    good_classes = [0, 1, 2, 3, 4, 5, 6]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "add_is_hit_cxi", OFFLINE_DIR / "add_is_hit_cxi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cxi_file = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi"

    is_hit = mod.add_is_hit_cxi(cxi_file, hit_score_threshold, size_min, size_max, good_classes)
    return is_hit

@Variable(title="Powder radial profile", data="proc")
def powder_radial_profile(run, run_no: "meta#run_number",
                          proposal_path: "meta#proposal_path",
                          good_hits_state: "var#good_hits"):
    """Azimuthal average of the good-hit powder vs momentum transfer q, with the
    beamline background (data_white x summed background_weighting over the good
    hits) subtracted, overlaid with the radial profile of the 3D reference model
    (scratch/models/Ery_075nm.h5) scaled to the powder. Depends on good_hits,
    which writes detector_1/powder. Shares one implementation with
    offline/add_powder_radial_profile.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "add_powder_radial_profile",
        OFFLINE_DIR / "add_powder_radial_profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cxi_file   = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi"
    model_file = proposal_path / "scratch" / "models" / "Ery_075nm.h5"

    q_p, P, raw, background, q_m, M = mod.add_powder_radial_profile(cxi_file, model_file)
    fig = mod.make_powder_profile_figure(q_p, P, raw, background, q_m, M, run_no)

    # drop a PNG next to the CXI for offline viewing/transfer
    out = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_powder_radial_profile.png"
    fig.savefig(out)
    return fig
