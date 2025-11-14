
# Step-by-step Docker Managing

Step-by-step guide on Docker Image managing operations: viz., build, run, execute, additional installations, save image, export image.

## Build & Run docker image
```
docker run --gpus all -it -v /home/jofre/Students/Nuria_Masclans/repositories/ODT/:/ODT/ --name ODT-RL nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

Alert! -> you should get a -devel- image for cuda compiling, do not install -base- or -runtime- versions. 

## Execute docker image
```
docker start ODT-RL
docker exec -it ODT-RL bash
```

## Additional installations
Run the following commands when executing the docker container for the first time:

- Install cmake:
```
apt-get update
apt-get install -y cmake
```

- Install c++ compiler:
```
apt-get install -y g++
export CXX=/usr/bin/g++
```

- Install git: 
```
apt-get install -y git
```

- Install doxygen:
```
apt-get install -y doxygen
```

- Install Boost library (ODT dependency):
```
apt-get install -y libboost-all-dev
```

- Install Yaml library (ODT dependency):
```
apt-get install -y libyaml-cpp-dev
```

- Install fmt library (ODT dependency):

Using ```apt-get```, the only available fmt library is ```libfmt-dev```, which installes a __shared__ library (bin files ```libfmt.so```), which will lead to compilation errors. C++ requires the __static__ library (bin files ```libfmt.a```).

As fmt static library is found by apt-get, the library is built from source code as:
```
cd /usr/src/
git clone --depth 1 https://github.com/fmtlib/fmt.git
cd fmt; mkdir build; cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE ..
```
This creates the required ```libfmt.a``` file in ```/usr/src/fmt/build```. 

To detect and use the installed version of ```fmt``` library, the lib and include paths are specified in the ```user_config_dockerContainer``` file as:
```
set(FMT_INCLUDE_DIR "/usr/src/fmt/include" CACHE PATH "fmt include location")
set(FMT_LIB_DIR "/usr/src/fmt/build" CACHE PATH "fmt library location")
```

- Install Cantera library (ODT dependency) - Cantera PPA:
```
apt install -y software-properties-common
apt-add-repository ppa:cantera-team/cantera
apt install -y cantera-python3 libcantera-dev
```

## Save Docker Container as Docker Image

The previous docker container is saved as an image to be exported in the next step, so that we do not need to follow the previous installation steps every time we use a different node. To save the docker container as a docker image:

```
$ docker commit <container> <image>
```
In this case: 
```
$ docker ps -a
CONTAINER ID   IMAGE                                                 COMMAND                  CREATED       STATUS         PORTS     NAMES
c34c54a93062   nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04           "/opt/nvidia/nvidia_…"   2 hours ago   Up 5 minutes             ODT-RL

$ docker commit c34c54a93062 nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL
```

Now, we have the original image (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`) and the just created image with all required packages installed (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL`):
```
$ docker images
REPOSITORY                  TAG                                          IMAGE ID       CREATED         SIZE
nvidia/cuda                 12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL   6b5487fadc1a   6 seconds ago   11.5GB
nvidia/cuda                 12.1.0-cudnn8-devel-ubuntu22.04              3d89d59a4dc1   4 weeks ago     9.49GB
```

## Export Image 

If direct access to the Docker registry isn't feasible (perhaps due to firewall restrictions or no direct internet access), you can save your image as a tar archive and transfer it manually.
This is recomended on this project because the docker image `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` is scheduled to be deleted. 

It is exported both the original `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` and the working image container with all necessary libraries installed `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL`:

On the source server:
```
docker save <image_tag> -o <file_name>
```
in my case:
```
docker save nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 -o nvidia-cuda-12.1.0-cudnn8-devel-ubuntu22.04.tar
docker save nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL -o nvidia-cuda-12.1.0-cudnn8-devel-ubuntu22.04-for-ODT-RL.tar
```

Then, transfer the generated jax_23.08_paxml_py3_for_PI_DeepONets.tar file to the target server using any preferred method (SCP, FTP, etc.).

On the target server:
```
docker load -i <file_name>
```
This method doesn't require direct access to the Docker registry but involves manual file transfer.