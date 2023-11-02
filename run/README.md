## Run directory

Run the ODT code from this directory, which contains the executable. There are several run scripts for running in different modes:
* **runOneRlz.sh**: run a single ODT realization,
* **runManyRlz.sh**: run several realizations in succession on a single processor,
* **slrmJob.sh**: slurm submission script for parallel simulations,
* **slrmJob_array.sh** slurm submission script for parallel simulations using the slurm *array* feature.

For each script, edit the input file being used and provide an arbitrary output case name.

### Run one realization of a channel flow case

From this location, run in terminal: 
```
cd ../build; make clean; cmake -C user_config ../source; make -j8; cd ../run; ./runOneRlz.sh <case_name>;
```