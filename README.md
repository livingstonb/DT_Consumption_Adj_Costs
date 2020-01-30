# Discrete Time Model with Consumption Adjustment Costs

The code in this repository was written by Brian Livingston
(livingstonb@uchicago.edu).
The spline interpolation algorithm was taken from *Numerical Recipes in C*,
Columbia University Press 1992,
by Press, Teukolsky, Vetterling, and Flannery.

## Getting Started

This repository has been used successfully in both Unix and Linux.
Adjustments may be necessary to compile and execute the code in Windows.

### Prerequisites

The user must have Python 3 with the following packages:

* matplotlib
* scipy
* itertools
* pandas
* numpy
* distutils
* Cython

One can use the Anaconda
or Miniconda package manager to install the above prerequesites
without too much trouble.
GNU make and gcc are also required.


### Compiling with Cython

In the terminal, navigate to the main directory, which should include the makefile,
and execute the *make* command.
If problems are encountered here, the user may want to check the *setup.py*
file which executes the compiling steps via *distutils* and *Cython*.
The *make clean* command will delete all files created by the *make* command.
This is occasionally necessary if the *Cython* .pyx and/or .pxd files have been changed
and the compiler produces an unexpected error during an attempt to re-compile.

## Replicating the experiments in Fuster, Kaplan, and Zafar (2020)

The script *master_replication.py* is provided to solve for the calibrations
presented in the paper and compute statistics.
The income process used for these calibrations is contained in the MATLAB file
*quarterly_b.mat*.

### The calibration

The desired calibration must be selected prior to running the code.
This is done in the *CHOOSE CALIBRATION* section in *master_replication.py*.
The parameters specific to each calibration can be viewed in the subsequent
sections of the script.
Note than any parameters not explicitly set in *master_replication.py*
will take their default values, which are declared in *Params.pyx*.

### Calibrating parameters to match the data

As discussed in the paper, we've selected parameters in part to match
statistics observed in the data (e.g. mean wealth equal to 3.2 times mean annual income).
The *master_replication.py* script omits those calibration routines, which
involved solving the model iteratively using a *scipy* root-finding algorithm.
To recreate those routines, see *master.py* for a taste of how
this was done.

### Output

The output tables are saved as csv's in the *output* directory.