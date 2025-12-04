export CC=/public/home/jzc/gcc-14.2.0/build/bin/gcc
export CXX=/public/home/jzc/gcc-14.2.0/build/bin/g++
export LIBRARY_PATH=/public/home/jzc/gcc-14.2.0/build/lib:/public/home/jzc/gcc-14.2.0/build/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/jzc/gcc-14.2.0/build/lib:/public/home/jzc/gcc-14.2.0/build/lib64:$LD_LIBRARY_PATH

export PATH=/public/home/jzc/mpich-4.2.3/build/bin:$PATH

# Only for test
mpirun -np 32 -host mn4:32 ./build/bin/mpi_distributed_time_test 512
mpirun -np 32 -host mn4:16,mn5:16 ./build/bin/mpi_distributed_time_test 512