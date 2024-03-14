# Functions for creating and running a load-shifting optimisation model. 

import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, CONTINUOUS, BINARY


# Flow:
# Take in existing gen profile (hybridised, synthesised or as is) and optimisation
# inputs. 

# Create variables
# Create model
# Add constraints
# Run model, get outputs

# Output new load profile as time-stamped df

# Some helper functions needed:
# 1. Function to get the length of time interval used in the run through
# 2. Testing / input validation functions to make sure inputs are formatted 
#    correctly, right length, etc. 

def validate_profile(gen_profile):
    
    return

