# Distributed-NN
A distributed Neural Network optimizations applied with synchronous and asynchronous approaches and ring all-reduce implementation

Running tests: <br/>
1. Allreduce test: running allreduce_test.py with an amount of cores, an amount of processes using mpi example (on slurm): srun -K -c 4 -n 4 --mpi=pmi2 --pty python3 allreduce_test.py
2. Synchronous test example: srun -K -c 4 -n 4 --mpi=pmi2 --pty python3 main.py sync
3. Asynchronous test example: srun -K -c 4 -n 4 --mpi=pmi2 --pty python3 main.py 2 (2 is the number of master processes, the rest will be workers)
