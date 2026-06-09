# xfel10662
3D Diffractive Imaging of Single-Protein with Hard X-ray Laser at the SFX/SPB instrument

## Links
- [zulip run logs](https://mylog.connect.xfel.eu/#narrow/channel/982-SPB_10662_HenryChapman)
- [damnit table](https://damnit.xfel.eu/app/proposal/10662)
- [discord](https://discord.gg/9fhycZz6)

## Files
```text
Experiment root  : /gpfs/exfel/exp/SPB/202601/p010662/
AGIPD Data       :   /raw/r*/RAW*AGIPD*.h5
Meta  Data       :   /raw/r*/RAW*DA*.h5
AGIPD (corrected):   /proc/r*/COR*AGIPD*.h5
This repo        :   /usr/Shared/xfel10662/
Output files     :   /scratch/
```

## Pipeline
```text
XFEL Data (/raw)
   └► facility calibrations (/proc/r*/CORR*)
     └► vds files  (/scratch/vds/r*.cxi)
       └► cxi file with hits only (/scratch/saved_hits/r*_hits.cxi)
         ├► classification (cxi file)
         ├► sizing (cxi file)
```

## Scripts
These are located in `offline/`
```text
XFEL Data (/raw)
   └► facility calibrations (automatically triggered by EuXFEL)
      └► VDS files (submit_vds.sh)
        └► cxi files for hits (submit_cxi.sh)
          ├► fit to precalculated pdb simulations (submit_classification.sh)
          ├► fit scale factors to pdb simulations (submit_sizing.sh)
          ├► add "is_hit" based on above (add_is_hit_cxi.py)
```

## DAMNIT
[damnit docs](https://damnit.xfel.eu)

This is a tool developed by the EuXFEL to automatically log run information in a spread sheet. It also automatically launches analysis scripts in this repo once preconditions are met.

The cells in the spread sheet are determined by the `/DAMNIT/context.py` file.


## Example cxi file
```bash
% h5ls -r /gpfs/exfel/exp/SPB/202601/p010662/scratch/saved_hits/r0600_hits_test.cxi
/                        Group
/cxi_version             Dataset {SCALAR}
/entry_1                 Group
/entry_1/data_1          Group
/entry_1/data_1/data     Soft Link {/entry_1/instrument_1/detector_1/data}
/entry_1/experiment_identifier Dataset {461}
/entry_1/instrument_1    Group
/entry_1/instrument_1/detector_1 Group
/entry_1/instrument_1/detector_1/background_weighting Dataset {461}
/entry_1/instrument_1/detector_1/basis_vectors Dataset {16, 2, 3}
/entry_1/instrument_1/detector_1/cellId Dataset {461}
/entry_1/instrument_1/detector_1/corner_position Dataset {16, 3}
/entry_1/instrument_1/detector_1/data Dataset {461, 16, 512, 128}
/entry_1/instrument_1/detector_1/data_white Dataset {16, 512, 128}
/entry_1/instrument_1/detector_1/description Dataset {SCALAR}
/entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
/entry_1/instrument_1/detector_1/experiment_identifier Soft Link {/entry_1/experiment_identifier}
/entry_1/instrument_1/detector_1/mask Dataset {16, 512, 128}
/entry_1/instrument_1/detector_1/module_identifier Dataset {16}
/entry_1/instrument_1/detector_1/score Group
/entry_1/instrument_1/detector_1/score/hit_score Dataset {461}
/entry_1/instrument_1/detector_1/score/hit_sigma Dataset {461}
/entry_1/instrument_1/detector_1/score/is_hit Dataset {461}
/entry_1/instrument_1/detector_1/score/photon_counts Dataset {461}
/entry_1/instrument_1/detector_1/trainId Dataset {461}
/entry_1/instrument_1/detector_1/vds_index Dataset {461}
/entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
/entry_1/instrument_1/detector_1/xyz_map Dataset {3, 16, 512, 128}
/entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
/entry_1/instrument_1/electrospray Group
/entry_1/instrument_1/electrospray/CO2_capillary Dataset {461}
/entry_1/instrument_1/electrospray/CO2_chamber Dataset {461}
/entry_1/instrument_1/electrospray/DP2 Dataset {461}
/entry_1/instrument_1/electrospray/He_chamber Dataset {461}
/entry_1/instrument_1/electrospray/N2_capillary Dataset {461}
/entry_1/instrument_1/electrospray/N2_chamber Dataset {461}
/entry_1/instrument_1/electrospray/current Dataset {461}
/entry_1/instrument_1/electrospray/experiment_identifier Soft Link {/entry_1/experiment_identifier}
/entry_1/instrument_1/electrospray/liquidjet_He Dataset {461}
/entry_1/instrument_1/electrospray/voltage Dataset {461}
/entry_1/instrument_1/name Dataset {SCALAR}
/entry_1/instrument_1/source_1 Group
/entry_1/instrument_1/source_1/energy Dataset {461}
/entry_1/instrument_1/source_1/name Dataset {SCALAR}
/entry_1/instrument_1/source_1/pulse_energy Dataset {461}
/entry_1/result_1        Group
/entry_1/result_1/class_labels Dataset {13}
/entry_1/result_1/data   Dataset {461}
/entry_1/result_1/description Dataset {SCALAR}
/entry_1/result_1/detector_1 Soft Link {/entry_1/instrument_1/detector_1}
/entry_1/result_1/frames Dataset {151}
/entry_1/result_1/process_1 Group
/entry_1/result_1/process_1/command Dataset {SCALAR}
/entry_1/result_1/process_1/date Dataset {SCALAR}
/entry_1/result_1/process_1/program Dataset {SCALAR}
/entry_1/result_1/process_1/version Dataset {SCALAR}
/entry_1/result_2        Group
/entry_1/result_2/class_index Dataset {461}
/entry_1/result_2/data   Dataset {461, 3}
/entry_1/result_2/description Dataset {SCALAR}
/entry_1/result_2/detector_1 Soft Link {/entry_1/instrument_1/detector_1}
/entry_1/result_2/frames Dataset {151}
/entry_1/result_2/process_1 Group
/entry_1/result_2/process_1/command Dataset {SCALAR}
/entry_1/result_2/process_1/date Dataset {SCALAR}
/entry_1/result_2/process_1/program Dataset {SCALAR}
/entry_1/result_2/process_1/version Dataset {SCALAR}
/entry_1/result_2/scale_index Dataset {461}
/entry_1/result_2/scale_unique Dataset {1, 289, 3}
/entry_1/sample_1        Group
/entry_1/sample_1/name   Dataset {SCALAR}
/entry_1/start_time      Dataset {SCALAR}
/entry_1/title           Dataset {SCALAR}
```
