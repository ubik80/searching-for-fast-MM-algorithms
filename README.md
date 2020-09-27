# searching-for-fast-MM-algorithms
searching for fast MM algorithms / masters thesis

read:

https://www.researchgate.net/publication/301874145_A_Network_That_Learns_Strassen_Multiplication

https://www.researchgate.net/publication/341942316_Searching_for_fast_matrix_multiplication_algorithms

The subject of this master's thesis is an example of automated learning of algorithms.
Bilinear algorithms with fewer than n^3 products for the multiplication of n x n matrices
are searched for. We are using a combination of the backpropagation algorithm and
projection algorithms, the Difference-Map Algorithm and, alternatively, another heuristic
algorithm. Bilinear matrix multiplication algorithms can be represented in network
form, which motivates the use of the backpropagation algorithm. The solutions found
by training the network contain much more than n^3 products, the set of solutions needs
to be constraint to integer solutions only containing -1, 0 or 1 for the weights of the
network. Parameters for the algorithms are determined through tests. For better performance
of the search, an initialization step is developed. For reducing also additions and
subtractions in found matrix multiplication algorithms, another algorithm that exploits
common expressions is developed. The combination of the backpropagation algorithm
and the Difference-Map Algorithm is tested for the cases n = 2, n = 3 and n = 5.
Several thousand solutions are found for the case n = 3, among these solutions, the one
with the least number of additions and subtractions is selected and shortly discussed.
The behaviour of the Difference-Map Algorithm applied to our problem is investigated.

The implementation was mainly done in python (python 3.7). For some parts of the
code, a just-in-time compiler was used (numba). The backpropagation algorithm, that
consumes most of the running time, was implemented as a C++ extension (with pybind11). 
Even so, it is obvious that the backpropagation algorithm is the most time
consuming part of the whole solution, that assumption was confirmed using a profiler
(line profiler). To shorten the running time, the solutions is implemented to run on
multiple cores (multiprocessing package for python).
