# Data Visualization
import matplotlib.pyplot as plt  # Library for creating plots and visualizations
import numpy as np  # Library for numerical computing
import pandas as pd  # Library for data manipulation and analysis
import seaborn as sns  # Library for creating plots and visualizations

# Reading education data into Python from a CSV file
education = pd.read_csv(r"C:\Data\education.csv")

# Checking the shape of the education data (number of rows and columns)
education.shape

# Creating a bar plot of GMAT scores
plt.bar(height=education.gmat, x=np.arange(1, 774, 1))  # Initializing the parameters

# Creating a histogram of GMAT scores with default parameters
plt.hist(education.gmat)  # histogram
#labeling title and x, Y 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/EDA/EDA 1/EDA-Datasets/education.csv')

# Create the histogram
plt.hist(df['workex'], bins=10, edgecolor='black')

# Set the x-axis label
plt.xlabel('Values')

# Set the y-axis label
plt.ylabel('Frequency')

# Set the title
plt.title('Histogram of Work Experience')

# Display the plot
plt.show()



# Creating a histogram of GMAT scores with specified bins, color, and edge color
plt.hist(education.gmat, bins=[600, 680, 710, 740, 780], color='green', edgecolor="red")

# Creating a histogram of work experience with default parameters
plt.hist(education.workex)

# Creating a histogram of work experience with specified color, edge color, and number of bins
plt.hist(education.workex, color='red', edgecolor="black", bins=6)

# Getting help on the hist function
help(plt.hist)

# Using Seaborn to create a histogram (deprecated)
sns.distplot(education.gmat)

# Using Seaborn to create a histogram
sns.displot(education.gmat)

# Creating a box plot of GMAT scores
plt.boxplot(education.gmat)
plt.figure()

# Getting help on the boxplot function
help(plt.boxplot)

# Creating a density plot of GMAT scores using Seaborn
sns.kdeplot(education.gmat)

# Creating a density plot of GMAT scores using Seaborn with specified bandwidth and filling the area under the curve
sns.kdeplot(education.gmat, bw=0.5, fill=True)

# Descriptive Statistics
# describe function will return descriptive statistics including the 
# central tendency, dispersion and shape of a dataset's distribution.

# Displaying descriptive statistics of the education data
education.describe()

# Bivariate visualization
# Scatter plot

# Reading car data into Python from a CSV file
cars = pd.read_csv("C:/Data/Cars.csv")

# Displaying information about the car data
cars.info()

# Creating a scatter plot of horsepower (HP) vs. miles per gallon (MPG)
plt.scatter(x=cars['HP'], y=cars['MPG']) 

# Creating a scatter plot of horsepower (HP) vs. sale price (SP) with green color
plt.scatter(x=cars['HP'], y=cars['SP'], color='green') 

