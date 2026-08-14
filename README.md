# ``relaxometrynmr``: NMR relaxometry made easy !

``relaxometrynmr`` an open-source Python package for solid-state NMR relaxation data analysis: T1, T1_rho and T2.

This package is only compatible with Bruker's NMR data (see User's guide).

# Why ``relaxometrynmr``?
This package is built for NMR relaxometry data analysis and offers a user-friendly data processing. It streamlines the analysis of relaxation time constant (T1, T1_rho or T2) and reduces analysis time by more than 50%. It comprises several built-in modeling functions such as mono-, di-, tri-, and stretch-exponential. These functions offer the flexibility to model a wide range of relaxation behaviors -- from simple to complex systems.



# Key Features

## Comprehensive NMR Data Handling
  - Supports mainly Bruker's data 
  - Handles data reading, conversion, and processing seamlessly
  - Automatically detects and loads delay list (vdlist, vplist, or vclist) for relaxometry experiments
## Advanced Data Processing
  - Zero-filling for improved spectral resolution
  - Phase correction (0th and 1st order) for signal representation in pure-absorption mode
  - Gaussian apodisation for line broadening and improvement of signal-noise ratio
## Spectral Integration
 - Numerical integration methods (trapezoid and Simpson's rule) for robust quantification of peak area
## Visualisation
  - Full spectrum view for context
  - Zoomed-in region for detailed analysis
## Modelling and Data Fitting
  - multiple-component models: mono-, bi-, and tri-exponential functions for simple to complex relaxation
  - stretched exponential models: for systems with non-standard relaxation dynamics (e.g., disordered materials)
  - decay analysis: tools for analysing exponential decay curves with multiple components
  
# Install

```bash
pip install relaxometrynmr

```

# Dependencies

The following packages are required:

```bash

nmrglue 
numpy
matplotlib >= 3.9.0
mrsimulator = 1.0

```

You can install these dependencies using pip:

```bash

pip install nmrglue numpy scipy mrsimulator = 1.0

```

Examples can be found in User Guide



# Contact

For questions and support, please open an issue in the GitHub repository.