2 + 2 # Function F9
# Works as calculator

# Python Libraries (Packages)
# pip install <package name> - To install library (package), execute the code in Command prompt
# pip install pandas

import pandas as pd

# List all the attributes and methods of the pandas module
dir(pd)

# Read data into Python from a CSV file
# Read the CSV file into the 'education' DataFrame using the specified file path
education = pd.read_csv(r"C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/EDA/EDA 1/EDA-Datasets/education.csv")

# Read the CSV file into the 'Education' DataFrame using the specified file path
Education = pd.read_csv("C:/Users/education.csv")

# Declare some variables
A = 10
a = 10.1

# Display information about the DataFrame "education"
education.info()


# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
# Calculate the mean of the 'workex' column in the 'education' DataFrame
education.workex.mean()

# Calculate the median of the 'workex' column in the 'education' DataFrame
education.workex.median()

# Calculate the mode of the 'workex' column in the 'education' DataFrame
education.workex.mode()

# Measures of Dispersion / Second moment business decision
education.workex.var() # variance
education.workex.std() # standard deviation
range = max(education.workex) - min(education.workex) # range
range


# Third moment business decision
# Calculate the skewness of the 'workex' column in the 'education' DataFrame
education.workex.skew()

# Calculate the skewness of the 'gmat' column in the 'education' DataFrame
education.gmat.skew()

# Fourth moment business decision
# Calculate the kurtosis of the 'workex' column in the 'education' DataFrame
education.workex.kurt()

