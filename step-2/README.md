# Using the FROSch - Geometric Preconditioner inside of deal.II
In this example step, we demonstrate the use of the geometric one-level preconditioner inside of deal.II 
on the example of Laplace's problem.
This example is based on the deal.II tutorial step-40. So, if you want to learn more about deal.II itself
or the Laplace equation, see https://www.dealii.org/current/doxygen/deal.II/step_40.html

## Running this example
1. Ensure that you installed deal.II and Trilinos as described in the root of this repository.

2. Download this repository
```
git clone git@github.com:kinnewig/dealii-FROSch-interface.git
```

3. Change into this example
```
cd dealii-FROSch-interface/step-2
```

4. Compile the example via CMake:
```
cmake -S . -B build
cmake --build build
```

5. (Optional: Switch to release)
By default the debugging modus is enabled, to switch to the release modus use:
```
cd build
make release
cd ..
```

6. Run the program
To execute the program:
```
mpirun -np <n_ranks> /build/step-2
```
remember to replace <n_ranks> with the number of ranks you want to use to execute this program.

## Modifying the parameters
In the context of domain decomposition one of the most interesting parameters to toy around is 
the number of subdomains. FROSch creates as many subdomains as there are ranks. So, to modify the 
amount of subdomains, you have to modify the number of ranks.

Moreover, you can modify many parameters directly in the step-1.xml file without recompiling any 
time you modify some parameters.
