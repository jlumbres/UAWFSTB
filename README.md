# UAWFSTB: Fresnel propagation simulation of UA Wavefront Testbed setup
This directory contains various notebooks which examine the Fresnel propagation effects for the UA Wavefront Testbed setup. It is the testbed used for analyzing the vAPP mask that will be inserted into the MagAO-X system. The primary objective for this project is to obtain a Fresnel propagation result for the testbed with the vAPP inserted. By implementing the various optical elements' surface values and calculating the propagation with Fresnel diffraction, this project aims to provide simulated support for testing other developing technologies, such as LDFC.

Code written by Jennifer Lumbres with generous input from Kelsey Miller.

## Requirements and Installation
### <i>Prerequisites</i>:
- Python 3 (all code is documented using Jupyter Notebook)
- numpy, scipy, matplotlib, astropy, etc 
- POPPY (Download and install from here: https://github.com/mperrin/poppy)

For installing Python3, Jupyter Notebook, and the required libraries, I strongly recommend using Anaconda3: https://www.continuum.io/downloads

## Getting Started
After installing the prerequisites, you can download and run any of the notebooks as you need. The notebooks are independent of each other but all reference to the same data folder, which contains the various PSD and mask FITS files used. Those files don't exist yet, primarily because they aren't made...

Currently, the simulation sampling and oversampling has changed to 256 pix sampling and 8x oversampling.

Please note that different notebooks will have different setups.

## Warnings, Disclaimers
A lot of the code may have walls of warnings. They are not detrimental to the operation. Later notebooks have "silenced" most of the wall of warnings.

The bare minimum files are posted. If there is a file missing in the repository that is mandatory for calculating the science PSFs, please contact me and I will upload it.

This is still in an organizing phase, so some code may be moved around occasionally. When this occurs, there will be a note posted somewhere here.

This code is only a Fresnel simulation for the UA Wavefront Testbed system. It is not comparable to a model produced by a high fidelity optical design software (Zemax, Code V, etc).
