#!/bin/bash
echo sourcing environment variables
source ../source_at_maxwell.sh
echo SANDBOX=$SANDBOX
echo REPO_ON_MAXWELL=$REPO_ON_MAXWELL

#echo sending repo
cd ..; ./sync_with_maxwell.sh

ssh max-exfl-display "cd $REPO_ON_MAXWELL; source source_at_maxwell.sh; cd DAMNIT; ./run_damnit.sh $1"
#ssh max-exfl-display "cd $REPO_ON_MAXWELL; source source_at_maxwell.sh; cd DAMNIT; pwd; ls"

