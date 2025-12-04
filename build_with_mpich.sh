export CC=/public/home/jzc/gcc-14.2.0/build/bin/gcc
export CXX=/public/home/jzc/gcc-14.2.0/build/bin/g++
export LIBRARY_PATH=/public/home/jzc/gcc-14.2.0/build/lib:/public/home/jzc/gcc-14.2.0/build/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/public/home/jzc/gcc-14.2.0/build/lib:/public/home/jzc/gcc-14.2.0/build/lib64:$LD_LIBRARY_PATH

export PATH=/public/home/jzc/mpich-4.2.3/build/bin:$PATH

cmake -S . -B build -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=PureMPI
cmake --build build --parallel $(nproc)