# ContrastFitterEBL
 Script for the import of `.csv` or `.xlsx` data to produce contrast curves and fitted contrast values.

## Setup and installation (local)
Clone this repository to the desired destination. To install dependencies and run, navigate to the folder called `ContrastFitterEBL` containing `requirements.txt` and run 

    pip install -r requirements.txt
    ./contrast_fits_GUI.py  

## Program details
Organise data in `.xlsx`, `.xls`, or `.csv`, with dose values in the first column and resist thickness values in subsequent columns. If the resist thickness values are not normalised, the program will normalise them. You can use the header columns to label the resist process, which will then be used as labels when plotting datasets. An example dataset can be found at `example_data.xlsx`.

Exported plots are saved in `/fitted_plots/` and data saved in Excel sheets in `/fitted_values/`.

The contrast curve describes the remaining resist fraction of a uniformly illuminated resist versus the logarithm of the applied exposure dose.

contrast $= \gamma = \frac{1}{\log_{10} \left( \frac{D_{100}}{D_0} \right)}$
- $D_{100}$ is the dose for FULL resist removal (linearised).
- $D_0$   is the dose for NO resist removal (linearised).

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

**IBM FITTING METHOD** (`fit_function='IBM'`)

Using the empirical technique by Leo Ocola (IBM, 2023). 

$\text{NRT} = C_0 - \exp[S * (D - D_c)]$. Then, contrast $= \gamma = \ln(10) * S * D_c$.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

**CMTF FITTING METHOD** (`fit_function='linear'`)

From Devin Brown's 2023 Georgia Tech presentation.

$\text{CMTF} = (D_{100} - D_0) / (D_{100} + D_0) = \frac{10^(1/\gamma) - 1}{10^(1/\gamma) + 1}$

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#### GUI built using Tkinter
https://docs.python.org/3/library/tkinter.html