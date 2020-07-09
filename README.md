# Discrete Time Model with Consumption Adjustment Costs

The code in this repository was written by Brian Livingston
(livingstonb@uchicago.edu).

## Getting Started

This repository has been used successfully in both Unix and Linux.
Adjustments may be necessary to compile and execute the code in Windows.

### Prerequisites

The user must have Python 3 with the following packages:

* matplotlib
* scipy
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

### The calibrations

The desired calibration must be selected prior to running the code.
This is done in the *CHOOSE REPLICATION* section in *master.py*.
Note than any parameters not explicitly set in *calibrations.py*
will take the default values declared in *Defaults.py*.

### Running custom calibrations

To run other calibrations, it is necessary to modify the code in which
the calibration dictionary is consructed, or write new code for that
purpose. The calibration dictionaries are created in *calibrations.py*.
To target a particular statistic via a numerical solver, I've written
the *Calibrator* class which allows one to set the variables to be varied,
bounds on those variables, and the target statistics. Note that varying the
adjustment cost to hit different MPC statistics can be tricky, I suggest
looking at the settings I used in *load_calibrations* in *calibrations.py*.

### Output

The output tables are saved as csv's in the *output* directory.