# Solving Time-Harmonic Maxwell's equations with deal.II and FROSch
This example builds on top of step-2. Instead of a Laplace problem, we consider the time-harmonic Maxwell equations here.

The time-harmonic Maxwell's equations are given by:
'''
     curl ( curl ( E ) - \omega^2 E = f(x)             on \Omega
     trace( E )                     = \trace (E_{inc}) on \Gamma_inc
'''

When you take a look into the source code, you will notice, that we dealii::shared::triangulation (instead of deal::distributed::triangulation as in step-2). 
Therefore, the full triangulation is stored on any rank. For lager problems this can become a bottleneck; however, in the current form of the implementation it greatly improves the performance.

## Running this example
1. Ensure that you installed deal.II and Trilinos as described in the root of this repository.

2. Download this repository
```
git clone git@github.com:kinnewig/dealii-FROSch-interface.git
```

3. Change into this example
```
cd dealii-FROSch-interface/step-maxwell
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
mpirun -np <n_ranks> /build/step-maxwell
```
remember to replace <n_ranks> with the number of ranks you want to use to execute this program.

## Modifying the parameters
In the context of domain decomposition one of the most interesting parameters to toy around is 
the number of subdomains. FROSch creates as many subdomains as there are ranks. So, to modify the 
amount of subdomains, you have to modify the number of ranks.

Moreover, you can modify many parameters directly in the step-1.xml file without recompiling any 
time you modify some parameters.
