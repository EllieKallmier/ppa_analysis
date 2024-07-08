# PPA Analysis

PPA Assessment Tool repo for the RACE for 2030 - 24/7 TRUZERO project. The tool provides functionality for optimising
a portfolio of renewable energy contracts, i.e. choosing what fraction of capacity to contract from set a generators in 
order to best match the combined generation profile with a load profile. Additionally, there is functionality for 
modelling battery operation, load flexibility, and calculating bills.

# Install

Create a Python new virtual environment and install the required dependencies, in the terminal:

1. Move to tool directory: ```cd /path/to/the/ppa_analysis/directory```
2. Create python virtual environment: ```python -m venv env```
3. Activate virtual environment:
   - windows: ```.\env\Scripts\activate```
   - mac/linux: ```source env/bin/activate```
4. Install dependencies: ```pip install -r requirements.txt```

# Notebook Interface

The tools capability can be explored and utilised through [interface.ipynb](interface.ipynb). 

1. After installing the tool, and with the virtual environment activated, launch jupyterlab using the terminal 
   command ```jupyter lab```. 
2. Then open the file interface.ipynb.
3. Then run each of notebooks cells in turn, and configure the inputs using the interactive input panels provided. 

# Loading further NEM data

The notebook [data_loading.ipynb](data_loading.ipynb)

# Bring your own load data

Add description of load data format required, and where to put data so the interface can find it.

# Documentation

## API documentation 

Provided in docstrings in the core tool modules, which are listed here along with the functionality each module 
provides:
    - ppa_analysis/import_data: preparing data
    - ppa_analysis/hybrid: contract portfolio optimisation
    - ppa_analysis/battery: battery optimisation to minimise cost of load not met by renewables
    - ppa_analysis/load_flex: load shifting optimisation to minimise cost of load not met by renewables
    - ppa_analysis/bill_calc: calculating the cost of procuring energy through the PPA

## Examples

In [example.ipynb](example.ipynb) a simple example using a single day of data demonstrates the tools functionality, 
similar to interface.ipynb, but without the input widgets so the user can see clearly how to use the tools through its 
Pyhton API.

## Glossary

The [glossary](glossary.md) provides definitions of key terms including contract types.

# Acknowledgements

Files msat_replicator.py, ppa.py, replicate_test.py, residuals.py, scenario_runner.py and tariffs.py were 
originally written by Nick Gorman as part of the MSAT-PPA Tool here: 
https://github.com/nick-gorman/MSAT-PPA-Python/tree/e9bef99ff914a826446f24697e983b16c23ced18
