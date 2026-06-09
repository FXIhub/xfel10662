#!/bin/bash
source ../source_at_maxwell.sh
source /etc/profile.d/modules.sh
module load exfel damnit/stable

cd $EXP_PREFIX/usr/Shared/amore/

# remove previous files
rm -f runs.sqlite
rm -rf extracted_data/

# initialise database
damnit init .

# Set a reservation
damnit db-config slurm_reservation $RESERVATION

# Set a partition
damnit db-config slurm_partition $PARTITION

# set time limit
damnit db-config slurm_time 05:00:00
