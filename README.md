# deal.II - FROSch Interface

This GitHub repository contains some code examples corresponding to the paper *Coupling deal.II and FROSch: A Sustainable and Accessible
(O)RAS Preconditioner*.

In step-1, a RAS preconditioner for the Laplace problem is demonstrated, and in in step-2, an ORAS preconditioner with Robin interface conditions is shown.

Important aspects of this work are part of deal.II, which you can find here: https://github.com/kinnewig/dealii/tree/FROSch-preconditioner.
Other important aspects of this work are part of FROSch, which is part of Trilinos. Therefore, you can find it here: https://github.com/kinnewig/Trilinos/tree/OptimizedSchwarz

## Citation
Please use the *Cite this repository* button in the *About* section of this repository.

## Installation - Dependencies
The wrapper for FROSch was not yet merged into deal.II. Also, the Geometric FROSch preconditioner has yet to be yet merged into Trilinos. 

### Method 1: Use the deal.II CMake Superbuild Script - 2 (dcs2)
This is a CMake script that installs dea.lII along with its dependencies. And it should be the easiest way to install deal.II with all its dependencies. For more details, see https://github.com/kinnewig/dcs2.

tl;dr:
1. Step: Download dcs2:
```
git clone git@github.com:kinnewig/dcs2.git
cd dcs2
```
2. Step: Run the install script:
```
./dcs.sh  -b </path/to/build> -p </path/to/install> --cmake-flags "-D TRILINOS_CUSTOM_URL=https://github.com/kinnewig/Trilinos.git -D TRILINOS_CUSTOM_TAG=OptimizedSchwarz -D DEALII_CUSTOM_URL=https://github.com/kinnewig/dealii.git -D DEALII_CUSTOM_TAG=FROSch-preconditioner"
```

Remember to replace `</path/to/build>` with the path where you would like to store the temporary files created while installing deal.II (the folder can be deleted once you successfully installed deal.II).

Also, remember to replace `</path/to/install>` with the folder where you would like to install deal.II.

If you have any problems feel free to open an issue on: https://github.com/kinnewig/dcs2/issues

### Method 2: Use the corresponding version of Trilinos and deal.II
1. Ensure that the dependencies for deal.II and Trilinos are installed!

2. Install Trilinos with the Geometric FORSch preconditioner:
```
git clone https://github.com/kinnewig/Trilinos.git
cd Trilinos
git checkout OptimizedSchwarz
```

3. Configure and install Trilinos, as usual.

4. Install deal.II with the FROSch wrapper:
```
git clone https://github.com/kinnewig/dealii.git
git checkout FROSch-preconditioner
```

5. Configure and install deal.II, as usual.

### Method 3: cherry-pick the FROSchWrapper and the  Geometric FORSch preconditioner.
1. Again, ensure that the dependencies for deal.II and Trilinos are installed!

2. Install the trillions version you like, e.g.:
```
git clone https://github.com/trilinos/Trilinos.git
```

3. cherry-pick the GeometricOverlappingOperator, the GeometricOneLevelPreconditioner, and the GeometricTwoLevelPreconditioner
```
git cherry-pick d82ab81ac6ef8f1c174cde8c911234c94b46f207
git cherry-pick b031ed306bdf3737d3a68de82423d0c5bfa0b65f
git cherry-pick 57c5f3d75b8322717ae8cf84a90e633c83d69862
```

4. Configure and install Trilinos, as usual.

5. Install the deal.II version you like, e.g.:
```
git clone https://github.com/dealii/dealii.git
```

6. Cherry-pick the Xpetra namespace and the FROSch wrapper:
```
git cherry-pick 6c4e3fb1bbe2178115b4503f98b2add26b34dfa1
git cherry-pick 8153feedd25b9e0ccf925db5aedbb20f70059df4
```

7. Configure and install deal.II, as usual.

## Run the examples
1. To run the examples, install deal.II and FROSch, as explained above.

2. Download this repository
```
git clone git@github.com:kinnewig/dealii-FROSch-interface.git
```

3. Change into the example you like to execute
```
cd dealii-FROSch-interface/step-<n>
```
remember to replace <n> with the example you like to run!

4. Compile the example via CMake:
```
cmake -S . -B build
cmake --build build
```

5. (Optional: Switch to release)
By default, the debugging modus is enabled. To switch to the release modus, use:
```
cd build
make release
cd ..
```

6. Run the program
To execute the program:
```
./build/step-<n>
```
again, remember to replace <n> with the example you like to run!
