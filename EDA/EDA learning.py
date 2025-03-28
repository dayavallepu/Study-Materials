# Q1) Calculate Mean, and Standard Deviation using Python code & draw inferences on the following data. Refer to the Datasets attachment for the data file.
# Hint: [Insights drawn from the data such as data is normally distributed/not, outliers, measures like mean, median, mode, variance, std. deviation]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\EDA\Assignment 3\Q1_a.csv")

# calulating the mean of speed column
mn=df.speed.mean()
print('mean of speed is :-',mn,'\n')

# Calculating the mean of distance column
mn2=df.dist.mean()
print('mean of distance is :-',mn2,'\n')

# calculating the median of speed 
mdn1=df.speed.median()
print('median of speed is ;-',mdn1,'\n')

# calculating the median of distance
mdn2=df.dist.median()
print('median of distance is ;-',mdn2,'\n')

# Calculating the mode of speed
md=df.speed.mode()
print('mode of speed :- ',md,'\n')

# Calculate the mode of distance

md1=df.dist.mode()
print('mode of distance is :- ',md1,'\n')

#Standard deviation of speed
s1=df.speed.std()
print('Standard deviation of speed :-',s1,'\n')
#Standard deviation of distance
s2=df.dist.std()
print('Standard deviation of distance :-',s2,'\n')
#  Variance of speed
print('variance of speed :- ',df.speed.var(),'\n')
# Variance of distance
print('variance of distance :- ',df.dist.var(),'\n')
#histogram of speed
plt.hist(df.speed)
plt.title('speed distribution')
plt.show()
#histogram of distance
plt.hist(df.dist,color='red',edgecolor='green',bins=20)
plt.title('distance distribution')
plt.show()

plt.boxplot(df.speed)
plt.title('speed Boxplot')
plt.show()

plt.boxplot(df.dist)
plt.title('distance Boxplot')
plt.show()

# Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, and Range & comment about the values / draw inferences, for the given dataset.
# -	For Points, Score, Weigh>
# Find Mean, Median, Mode, Variance, Standard Deviation, and Range and comment on the values/ Draw some inferences

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r'C:/Users/dayav/OneDrive/Desktop/360digiTMG/assignments 360/EDA/Assignment 3/Q7_EDA.xlsx')

#Calculating the mean value of Points
print(df.Points .mean()) # mean value = 3.596525

#Calculating the mean value of Score
print(df.Score.mean()) # mean value = 3.21156

#Calculating mean value of Weigh
print(df.Weigh.mean()) #mean value = 17.848

#Calculating the median value of Points
print(df.Points .median()) # meduan value = 3.695

#Calculating the median value of Score
print(df.Score.median()) # median value = 3.325

#Calculating median value of Weigh
print(df.Weigh.median()) #median value = 17.71

#Calculating the mode value of Points
print(df.Points .mode()) # mode value = 0    3.07
#                                       1    3.92

#Calculating the mode value of Score
print(df.Score.mode()) # mode value = 0  3.44

#Calculating mode value of Weigh
print(df.Weigh.mode()) #mode value = 0    17.02 ; 1    18.90

#calculating variance value of Points
print(df.Points.var()) #var value is 0.28588135080645166

#calculating variance value of Score
print(df.Score.var()) #var value is 0.9325025766129035

#calculating variance value of Weigh
print(df.Weigh.var()) #var value is 3.193166129032258

#calculating standard deviation value of Points
print(df.Points.std()) #std value is 0.5346787360709716

#calculating Standard deviation value of Score
print(df.Score.std()) #std value is 0.9656617299100672

#calculating standard dseviation value of Weigh
print(df.Weigh.std()) #std value is 1.7869432360968431

#calculating range of points 
print(max(df.Points) - min(df.Points)) # range of Points is 2.17

#calculating range of Score 
print(max(df.Score) - min(df.Score)) # range of Score is 3.832

#calculating range of Weigh 
print(max(df.Weigh) - min(df.Weigh)) # range of Weigh is 8.39

#calculating ditribution plot of Points
sns.distplot(df.Points) # for distribution plot we are using seaborn 
plt.title('distribution plot of points') # for the title purpose we used matplotlib
#calculating the box plot of Points
plt.boxplot(df.Points) # here we using boxplot for finding outlier presence 
plt.title('boxplot of Points') 

#calculating ditribution plot of Score
sns.distplot(df.Score ) # for distribution plot we are using seaborn 
plt.title('distribution plot of Score') # for the title purpose we used matplotlib
#calculating the box plot of Score
plt.boxplot(df.Score) # here we using boxplot for finding outlier presence 
plt.title('boxplot of Score') 

#calculating ditribution plot of Weigh
sns.distplot(df.Weigh) # for distribution plot we are using seaborn 
plt.title('distribution plot of Weigh') # for the title purpose we used matplotlib
#calculating the box plot of weigh
plt.boxplot(df.Weigh) # here we using boxplot for finding outlier presence 
plt.title('boxplot of Weigh')

# Q9) Look at the data given below. Plot the data, find the outliers, and find out:  μ,σ,σ^2
# Hint: [Use a plot that shows the data distribution, and skewness along with the outliers; also use Python code to evaluate measures of centrality and spread]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel(r'C:/Users/dayav/OneDrive/Desktop/360digiTMG/assignments 360/EDA/Assignment 3/Q9_EDA.xlsx')

#code for measures of central tendency

#Calculating the mean value of Measure_X
print(df.Measure_X.mean()) # mean value (μ) = 0.332713

#calculating the median value of Measure_X
print(df.Measure_X.median()) # meddian value = 0.2671

#calculating the mode value of Measure_X
print(df.Measure_X.mode())
# mode value = 
# 0     0.2414
# 1     0.2423
# 2     0.2439
# 3     0.2541
# 4     0.2553
# 5     0.2581
# 6     0.2599
# 7     0.2671
# 8     0.2825
# 9     0.2962
# 10    0.3295
# 11    0.3500
# 12    0.3942
# 13    0.4026
# 14    0.9136
#so here the mode values are more,so this is a multimodal we can say.
# calculating exactmode  point is mean of mode is 0.3327

# so here mean>median>mode so it is positively skewed
# will check the skewnes value
print(df.Measure_X.skew()) # the value is 3.2551132228850337 , if any value is greater than 0 that should be positively skewed


#measures of dispersion

#Calculating the std value of Measure_X
print(df.Measure_X.std()) # standard deviation value (σ) = 0.16945400921222029

#calculating variance value of Measure_X
print(df.Measure_X.var()) #variance value (σ^2) = 0.028714661238095233
# calculating rage here 
print(df['Measure_X'].max() - df['Measure_X'].min()) #range = 0.6721999999999999

# Identifying outliers using IQR
Q1 = df['Measure_X'].quantile(0.25)
Q3 = df['Measure_X'].quantile(0.75)
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR # lower bound value is 0.1271250000000001 
upper_bound = Q3 + 1.5 * IQR #upper bound value is 0.46732499999999993

outliers = df[(df['Measure_X'] < lower_bound) | (df['Measure_X'] > upper_bound)]
print("Outliers:",outliers) # the outlier is 10  Morgan Stanley     0.9136 this is upper outlier it is above than upper_bound


# we can see the plots here

#calculating ditribution plot of Measure_X
sns.distplot(df.Measure_X) # for distribution plot we are using seaborn 
plt.title('distribution plot of Measure_X') # for the title purpose we used matplotlib

#calculating the box plot of Measure_X
plt.boxplot(df.Measure_X) # here we using boxplot for finding outlier presence 
plt.title('boxplot of Measure_X') 








