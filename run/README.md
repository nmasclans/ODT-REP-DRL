## Run directory

Run the ODT code from this directory, which contains the executable. There are several run scripts for running in different modes:
* **runOneRlz.sh**: run a single ODT realization,
* **runManyRlz.sh**: run several realizations in succession on a single processor,
* **slrmJob.sh**: slurm submission script for parallel simulations,
* **slrmJob_array.sh** slurm submission script for parallel simulations using the slurm *array* feature.

For each script, edit the input file being used and provide an arbitrary output case name.

### Run one realization of a channel flow case

First, define `ODT_PATH` environment variable by running in bash terminal: 
```
export ODT_PATH=$(readlink -f ..)
```
You may add this path in the ./bashrc file.

Next, from this location, run in terminal: 
```
cd $ODT_PATH/build; make clean; cmake -C user_config ../source; make -j8; cd ../run; ./runOneRlz.sh <case_name> <rlz_number>;
```
In case you are compiling & running the code in a docker container (built as specified in main [README](./../README.md)), use the specific ```user_config_dockerContainer``` configuration:
```
cd $ODT_PATH/build; make clean; cmake -C user_config_dockerContainer ../source; make -j8; cd ../run; ./runOneRlz.sh <case_name> <rlz_number>;
```