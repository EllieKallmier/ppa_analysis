# PPA Analysis

PPA Assessment Tool repo for the RACE for 2030 - 24/7 TRUZERO project. The tool provides functionality for optimising
a portfolio of renewable energy contracts, i.e. choosing what fraction of capacity to contract from set a generators in 
order to best match the combined generation profile with a load profile. Additionally, there is functionality for 
modelling battery operation, load flexibility, and calculating bills.

# Install

Create a Python new virtual environment and install the required dependencies. In the terminal:

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

The notebook [data_loading.ipynb](data_loading.ipynb) allows users to load further years of NEM data through two tools developed by the Collaboration on Energy and Environmental Markets ([CEEM](https://www.ceem.unsw.edu.au)): 
- [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS), for fetching and formatting data from AEMO's MMS tables, and 
- [NEMED](https://github.com/UNSW-CEEM/NEMED), for processing grid emissions data.

# Bring your own load data

When using the interface.ipynb user's can use their own load data by adding a CSV containing the load data to the 
directory data_caches/c_and_i_customer_loads. Default expected format of the load data is one column named 'TS'
containing time stamps, and another named 'Load' containing the consumption in kWh for the period ending on the
time stamp. The expected format of the time stamp is day-month-year format, with various options like 
"DD-MM-YY HH:MM:SS" or "DD/MM/YY HH:MM:SS" being acceptable. The expected format can also be adjusted using the 
ppa_analysis/advanced_settings.py module.

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

In [api_examples.ipynb](api_examples.ipynb) a simple example using a single month of data demonstrates the tools 
functionality, similar to interface.ipynb, but without the input widgets so the user can see clearly how to use 
the tools through its Python API.

## Glossary

The [glossary](glossary.md) provides definitions of key terms including contract types.


# Acknowledgements

This tool was initially developed under the [RACE for 2030](https://racefor2030.com.au) project [24/7 TRUZERO](https://racefor2030.com.au/project/24-7-renewables-solutions-for-matching-tracking-and-enhancing-corporate-renewables-purchasing/), made possible by support from industry partners Enosi, AGL, Mirvac Ventures, Starling Energy Group and Buildings Alive.

Much of the functionality and sections of this tool were based on the [MSAT-PPA tool](https://github.com/nick-gorman/MSAT-PPA-Python/tree/e9bef99ff914a826446f24697e983b16c23ced18) developed by Nick Gorman.
