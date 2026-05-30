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

from damnit_ctx import Variable

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


HITFINDER_SOURCE = 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output'

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
    n_pulses = int(run[bunch_pattern, 'sase1.nPulses'].as_single_value())
    pulse_ids = run[bunch_pattern, 'sase1.pulseIds'].drop_empty_trains()[0].ndarray()[..., :n_pulses].squeeze()

    # compute repetition rate (in kHz) based on pulse IDs
    if n_pulses == 0:     
        rep_rate = 0.0
    elif n_pulses == 1:
        rep_rate = 1.0e-6
    else:
        pulseIdsIncr = np.gradient(pulse_ids)
        if np.all(pulseIdsIncr == pulseIdsIncr[0]):
            rep_rate = 1300. / 288. / pulseIdsIncr[0]
        else:
            rep_rate = np.nan
    return rep_rate

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
    return run[det_pos, 'encoderPosition'].drop_empty_trains()[0].ndarray()[0]
    
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

##@Variable(title="Taylor cone image")
##def taylor_cone_image(run):
##    return run[f"{taylor_cam}:daqOutput", 'data.image.pixels'].ndarray()[0]

# Electro spray

##@Variable(title="N2 flow in chamber (ls/min)", summary="mean")
##def n2_flow_chamber(run):
##    return run[n2_chm_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="CO2 flow in chamber (mln/min)", summary="mean")
##def co2_flow_chamber(run):
##    return run[co2_chm_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="He flow in chamber (ln/min)", summary="mean")
##def he_flow_chamber(run):
##    return run[he_chm_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="N2 flow in capillary (mln/min)", summary="mean")
##def n2_flow_capillary(run):
##    return run[n2_cap_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="CO2 flow in capillary (mln/min)", summary="mean")
##def co2_flow_capillary(run):
##    return run[co2_cap_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="dP: inner capillary (psi(g))", summary="mean")
##def dp_inner(run):
##    return run[dp_inner_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="dP: outer capillary (psi(a))", summary="mean")
##def dp_outer(run):
##    return run[dp_outer_flowmeter, "measureCapacity.value"].ndarray()

##@Variable(title="Electrospray current (A)", summary="mean")
##def esi_current(run):
##    return run[esi_hv, "channel1.current.value"].ndarray()

##@Variable(title="Electrospray voltage (V)", summary="mean")
##def esi_voltage(run):
##    return run[esi_hv, "channel1.voltage.value"].ndarray()

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
def spi_hit_rate(run):
    hit_arr = run[HITFINDER_SOURCE, 'data.hitFlag'].ndarray()
    n_frames = hit_arr.size
    n_hits = np.sum(hit_arr)
    return (n_hits/n_frames) * 100

@Variable(title="Num. hits", data="proc", summary="sum")
def num_hits(run):
    hit_arr = run[HITFINDER_SOURCE, 'data.hitFlag'].ndarray()
    tr_id_arr = run[HITFINDER_SOURCE, 'data.trainId'].ndarray()

    tr_ids, ix = np.unique(tr_id_arr, return_inverse=True)
    hits_per_train_arr = np.bincount(ix, weights=hit_arr, minlength=len(tr_ids))

    hits_per_train_xarr = xr.DataArray(
        hits_per_train_arr, dims=['trainId'], coords={'trainId': tr_ids}
    )

    return hits_per_train_xarr

@Variable(title="Hitscore", data="proc", summary="mean")
def spi_hitscore(run):
    hitscore = run[HITFINDER_SOURCE, 'data.hitscore'].ndarray()
    tid = run[HITFINDER_SOURCE, 'data.trainId'].ndarray()
    hitscore_xarr = xr.DataArray(
        hitscore, dims=['trainId'], coords={'trainId': tid}
    )
    return hitscore_xarr


### Offline pipeline triggers
#
# These variables shell out to the existing offline/submit_*.sh wrappers
# (which submit their own slurm job via `sbatch <<EOT ... EOT`), capture the
# resulting job id from the wrapper's stdout, then poll `sacct` until the job
# leaves the pending/running states. The final slurm state is returned as the
# cell value so it shows up in the DAMNIT table; any non-COMPLETED state
# raises so DAMNIT records the variable as errored.

_SLURM_BUSY = {
    "PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED",
    "RESV_DEL_HOLD", "SUSPENDED", "RUNNING", "COMPLETING", "STAGE_OUT",
    "RESIZING",
}


def _submit_and_wait(script_name, run_no, poll_s=30.0, timeout_s=6 * 3600):
    """Run offline/<script_name> <run_no>, parse the slurm job id, then poll
    sacct until it's no longer queued/running. Returns 'STATE (jobid)' on
    COMPLETED; raises on any other terminal state or on timeout."""
    script = OFFLINE_DIR / script_name
    if not script.exists():
        raise FileNotFoundError(script)

    sub = subprocess.run(["bash", str(script), str(run_no)],
                         capture_output=True, text=True, check=True)
    blob = (sub.stdout or "") + (sub.stderr or "")
    m = re.search(r"Submitted batch job (\d+)", blob)
    if not m:
        raise RuntimeError(f"no slurm job id in {script_name} output:\n{blob}")
    jobid = m.group(1)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sacct = subprocess.run(
            ["sacct", "-j", jobid, "--format=State",
             "--noheader", "--parsable2"],
            capture_output=True, text=True,
        )
        rows = [r.strip().rstrip("+") for r in sacct.stdout.splitlines() if r.strip()]
        if rows:
            state = rows[0].split()[0]
            if state not in _SLURM_BUSY:
                if state == "COMPLETED":
                    return f"{state} ({jobid})"
                raise RuntimeError(
                    f"slurm job {jobid} ({script_name}) ended in state {state}")
        time.sleep(poll_s)
    raise TimeoutError(
        f"slurm job {jobid} ({script_name}) still running after {timeout_s}s")


@Variable(title="VDS", data="proc", cluster=True)
def make_vds(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path"):
    """Build the per-run virtual CXI from proc data via offline/submit_vds.sh.
    Triggered once proc has been migrated (data='proc')."""
    state = _submit_and_wait("submit_vds.sh", run_no)
    vds = proposal_path / "scratch" / "vds" / f"r{run_no:04d}.cxi"
    if not vds.exists():
        raise RuntimeError(f"{state}: {vds} missing after submit_vds.sh")
    return state


@Variable(title="CXI", cluster=True)
def make_cxi(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
             vds_state: "var#make_vds"):
    """Build the hits CXI file (make_cxi_file.py + add_background_cxi.py in
    parallel, then a sidecar --merge) via offline/submit_cxi.sh. Waits for
    make_vds to finish first."""
    state = _submit_and_wait("submit_cxi.sh", run_no)
    cxi = proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi"
    if not cxi.exists():
        raise RuntimeError(f"{state}: {cxi} missing after submit_cxi.sh")
    return state


@Variable(title="VDS path")
def vds_path(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
             vds_state: "var#make_vds"):
    """Path to the virtual CXI written by make_vds. Depends on make_vds so the
    cell only populates once the file is on disk."""
    return str(proposal_path / "scratch" / "vds" / f"r{run_no:04d}.cxi")


@Variable(title="CXI path")
def cxi_path(run, run_no: "meta#run_number", proposal_path: "meta#proposal_path",
             cxi_state: "var#make_cxi"):
    """Path to the hits CXI written by make_cxi."""
    return str(proposal_path / "scratch" / "saved_hits" / f"r{run_no:04d}_hits.cxi")

@Variable(title="Classification", cluster=True)
def make_classification(run, run_no: "meta#run_number",
                        proposal_path: "meta#proposal_path",
                        cxi_state: "var#make_cxi"):
    """Run EMC classification on the hits CXI via offline/submit_classification.sh.
    Waits for make_cxi so the CXI file is on disk before submitting."""
    state = _submit_and_wait("submit_classification.sh", run_no)
    info = (proposal_path / "scratch" / "classification"
            / f"r{run_no:04d}" / "iteration_info.h5")
    if not info.exists():
        raise RuntimeError(
            f"{state}: {info} missing after submit_classification.sh")
    return state


@Variable(title="Class histogram")
def classification_histogram(run, run_no: "meta#run_number",
                             proposal_path: "meta#proposal_path",
                             class_state: "var#make_classification"):
    """Bar chart of most-likely class assignments. Loaded from
    offline/make_histogram.py so the CLI script and the DAMNIT cell share one
    implementation. Also drops a PNG next to iteration_info.h5 for offline
    viewing/transfer."""
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

### sizing

### peak intensity report

### background
