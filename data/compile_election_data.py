"""
This script is used to sync election data from its hosted location on Google
Drive, and compile into a master CSV file for AMOS to use to run analysis.

For efficiency reasons, this process is only run when new data is added to the
host location, and NOT if a simulation needs to be re ran, updates made to
AMOS source code, etc.

@author: Will Ferguson
@email: will.ferguson@teenpact.com || iamwillferguson@gmail.com || wferguson8@gatech.edu
@date: January 27, 2024
@license: None
"""