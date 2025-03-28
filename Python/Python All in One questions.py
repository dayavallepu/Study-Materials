#-1
a=list(input('enter the input '))
for i in range(1,len(a),1):
    print(a[i:]+a[:i])
    # print(a[:i])
    # print(a[i:])

2#regression
import re
l='mair'
k=re.search(r"m['ay']['ie']r",l)
print(k.group()) #group method will return the entire matched string

l = "mayer is the meier of the society"
matches = re.findall(r"m[ae][iy]er", l)

# Check if "meier" is in the list of matches and print it
if "meier" in matches:
    print("meier")
    
#3  selecting multisheets in a file
# import pandas as pd
# excelfile=r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\data analytics learning file.xlsx"
# sheet1=0
# sheet2= 1

# df_sheet1 = pd.read_excel(excelfile,sheet_name=sheet1)
# df_sheet2 = pd.read_excel(excelfile,sheet_name=sheet2)

# 24/07/24

#1. how do you perform list slicing in python?
# Ans:-
lis=[1,2.3,'data',('hii'),3,4,55,6] #here im taking multiple elements in a list and assighned in 'lis' variable
sli=lis[1:7:1]  #im applying here slicing inside square brackets im passing an start index : stop index : step index 
print(sli) # im printing the 'sli' varible ,in that im already performed a slicing



#2.what are lamba functions in python?
# ans:-
#add
l=lambda x: x+10
print(l(5))
#multiplication
l=lambda x,y :x*y
print(l(5,5))

# Using lambda with map to square each number in a list
numbers = [1, 2, 3, 4]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)

# Using lambda with filter to get even numbers from a list
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)

# from functools import reduce

#reduce 
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product) 

# sorted
pairs = [(1, 'one'), (2, 'two'), (3, 'three')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(sorted_pairs) 

# Using lambda to sort a dictionary by value
my_dict = {'a': 3, 'b': 1, 'c': 2}
sorted_dict = sorted(my_dict.items(), key=lambda item: item[1])
print(sorted_dict)

# max 
strings = ["short", "medium", "longest"]
longest_string = max(strings, key=lambda s: len(s))
print(longest_string)  

# min 
smallest_string = min(strings, key=lambda s: len(s))
print(smallest_string)  

#3.how do you read and write files in python
#read
f=open(r'Remember.txt','r')
f.seek(0) #file pointer change to read starting
print(f.read())
f.close

#write
f=open(r'Remember.txt','w')
f.write('im data scientist')
f.close()

#what are generators in python.
# Ans:- a generator is a function that produces or yield  a sequence of values using 'yield' method

#1)	Please take care of missing data present in the “Data.csv” file using python module 
#“sklearn.impute” and its methods, also collect all the data that has “Salary” less than “70,000”.

import pandas as pd
from sklearn.impute import SimpleImputer

data=pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\Python Assignments\python module 7\datasets\Data.csv")
#identifing the columns with missing values
print('Columns with missing values:-\n',data.isnull().sum())

# Impute missing values
# For simplicity, we'll use mean imputation for numerical columns and mode imputation for categorical columns

numerical_col=data.select_dtypes(include=['int64', 'float64']).columns
categorical_col=data.select_dtypes(include=['object']).columns

#Mean imputation for numerical columns
numerical_impu=SimpleImputer(strategy='mean')
data[numerical_col]=numerical_impu.fit_transform(data[numerical_col])
print(numerical_impu)
# Mode imputation for categorical columns
categorical_impu = SimpleImputer(strategy='most_frequent')
data[categorical_col] = categorical_impu.fit_transform(data[categorical_col])
print(categorical_impu)
#filtering the data based on salary < 70000
filt_data=data[data['Salaries']<70000]
print(filt_data)






#1)	Please take care of missing data present in the “Data.csv” file using python module 
#“sklearn.impute” and its methods, also collect all the data that has “Salary” less than “70,000”.

import pandas as pd
from sklearn.impute import SimpleImputer

data=pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\Python Assignments\python module 7\datasets\Data.csv")
#identifing the columns with missing values
print('Columns with missing values:-\n',data.isnull().sum())

# Impute missing values
# For simplicity, we'll use mean imputation for numerical columns and mode imputation for categorical columns

numerical_col=data.select_dtypes(include=['int64', 'float64']).columns
categorical_col=data.select_dtypes(include=['object']).columns

#Mean imputation for numerical columns
numerical_impu=SimpleImputer(strategy='mean')
data[numerical_col]=numerical_impu.fit_transform(data[numerical_col])
print(numerical_impu)
# Mode imputation for categorical columns
categorical_impu = SimpleImputer(strategy='most_frequent')
data[categorical_col] = categorical_impu.fit_transform(data[categorical_col])
print(categorical_impu)
#filtering the data based on salary < 70000
filt_data=data[data['Salaries']<70000]
print(filt_data)

# 2)	Subtracting dates: 
# Python date objects let us treat calendar dates as something similar to numbers: we can compare them, sort them, add, and even subtract them. Do math with dates in a way that would be a pain to do by hand. The 2007 Florida hurricane season was one of the busiest on record, with 8 hurricanes in one year. The first one hit on May 9th, 2007, and the last one hit on December 13th, 2007. How many days elapsed between the first and last hurricane in 2007?
# 	Instructions:
# 	Import date from datetime.
# 	Create a date object for May 9th, 2007, and assign it to the start variable.
# 	Create a date object for December 13th, 2007, and assign it to the end variable.
# 	Subtract start from end, to print the number of days in the resulting timedelta object.


from datetime import date

# Create a date object for May 9th, 2007, and assign it to the start variable
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007, and assign it to the end variable
end = date(2007, 12, 13)

# Subtract start from end to get the timedelta object
delta = end - start

# Print the number of days
print(f"Number of days between {start} and {end}: {delta.days} days")


# 3)	Representing dates in different ways
# Date objects in Python have a great number of ways they can be printed out as strings. In some cases, you want to know the date in a clear, language-agnostic format. In other cases, you want something which can fit into a paragraph and flow naturally.
# Print out the same date, August 26, 1992 (the day that Hurricane Andrew made landfall in Florida), in a number of different ways, by using the “ .strftime() ” method. Store it in a variable called “Andrew”. 
# Instructions: 	
# Print it in the format 'YYYY-MM', 'YYYY-DDD' and 'MONTH (YYYY)'

from datetime import date

# Create a date object for August 26, 1992
Andrew = date(1992, 8, 26)

# Print the date in different formats using the strftime() method
# Format: 'YYYY-MM'
format1 = Andrew.strftime('%Y-%m')

# Format: 'YYYY-DDD'
format2 = Andrew.strftime('%Y-%j')

# Format: 'MONTH (YYYY)'
format3 = Andrew.strftime('%B (%Y)')

print(f"Format 'YYYY-MM': {format1}")
print(f"Format 'YYYY-DDD': {format2}")
print(f"Format 'MONTH (YYYY)': {format3}")

# 4)	For the dataset “Indian_cities”, 
# a)	Find out top 10 states in female-male sex ratio
# b)	Find out top 10 cities in total number of graduates
# c)	Find out top 10 cities and their locations in respect of  total effective_literacy_rate.

import pandas as pd

# Load the dataset
df = pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\Python Assignments\python module 7\datasets\Indian_cities.csv")

# a) Find out top 10 states in female-male sex ratio

sex_ratio = df.groupby('state_name')['sex_ratio'].mean().sort_values(ascending=False).head(10)
print("Top 10 states in female-male sex ratio:")
print(sex_ratio)

# b) Find out top 10 cities in total number of graduates

cities_graduates = df[['name_of_city', 'total_graduates']].sort_values(by='total_graduates', ascending=False).head(10)
print("\nTop 10 cities in total number of graduates:")
print(cities_graduates)

# c) Find out top 10 cities and their locations in respect of total effective_literacy_rate

literacy_rate = df[['name_of_city', 'location', 'literates_total']].sort_values(by='literates_total', ascending=False).head(10)
print("\nTop 10 cities and their locations in respect of total effective literacy rate:")
print(literacy_rate)

# 5)	 For the data set “Indian_cities”
# a)	Construct histogram on literates_total and comment about the inferences
# b)	Construct scatter  plot between  male graduates and female graduates


# a)	Construct histogram on literates_total and comment about the inferences
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\Python Assignments\python module 7\datasets\Indian_cities.csv")

plt.figure(figsize=(10 ,6))
plt.hist(df['literates_total'], bins=50, edgecolor='black', alpha=0.8)
plt.title('Histogram of Total literates')
plt.xlabel('number of literates')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# b)	Construct scatter  plot between  male graduates and female graduates

plt.figure(figsize=(15, 9))
plt.scatter(df['male_graduates'], df['female_graduates'], alpha=0.7, edgecolor='w', s=100)
plt.title('Scatter Plot of Male Graduates vs Female Graduates')
plt.xlabel('Number of Male Graduates')
plt.ylabel('Number of Female Graduates')
plt.grid(True)
plt.show()



# For the data set “Indian_cities”
# a)	Construct Boxplot on total effective literacy rate and draw inferences
# b)	Find out the number of null values in each column of the dataset and delete them.

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\dayav\OneDrive\Desktop\360digiTMG\assignments 360\Python Assignments\python module 7\datasets\Indian_cities.csv")


# a) Construct boxplot for total effective literacy rate
plt.figure(figsize=(10, 6))
plt.boxplot(df['effective_literacy_rate_total'].dropna(), vert=False)
plt.title('Boxplot of Total Effective Literacy Rate')
plt.xlabel('Effective Literacy Rate')
plt.grid(True)
plt.show()

# b) Find out the number of null values in each column and delete them
null_counts = df.isnull().sum()
print("Number of null values in each column:")
print(null_counts)

# Drop rows with any null values
df_cleaned = df.dropna()

# Verify that there are no more null values
null_counts_after = df_cleaned.isnull().sum()
print("\nNumber of null values in each column after dropping:")
print(null_counts_after)

# 1.write a python function to calculate the 
 # factorial of a given number using both recursion and iteration.



def factorial_recursive(n):
    # Base case: factorial of 0 or 1 is 1
    if n == 0 or n == 1:
        return 1
    # Recursive case
    return n * factorial_recursive(n - 1)

# Example usage:
print(factorial_recursive(int(input('enter the number:-')))) 


def factorial_iterative(n):
    # Initialize the result to 1
    result = 1
    # Multiply result by every number from 2 to n
    for i in range(2, n + 1):
        result *= i
    return result

# Example usage:
print(factorial_iterative(int(input('enter the number:-'))))

# 2.write a python function that takes a list of integers and returns a new list with duplicate elements removed?
def remove_duplicates(input_list):
    # Create a set from the input list to remove duplicates
    unique_elements = set(input_list)
    # Convert the set back to a list
    return list(unique_elements)

# Example usage:
my_list = [1, 2, 2, 3, 4, 4, 5]
print(remove_duplicates(my_list))

# 3.write a python function to implement binary search on a sorted list of integers? The function should return the index of the target value if found ,or ‘-1’ if not
def binary_search(sorted_list, target):
    left, right = 0, len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return mid
        if sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
sorted_list = [1, 3, 5, 7, 9, 11]
target = 7
print(binary_search(sorted_list, target))

target = 4
print(binary_search(sorted_list, target)) 


# 4.  write a python function to merge two sorted lists into a single sorted list
def merge_sorted_lists(list1, list2):
    merged_list = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1

    # Append remaining elements, if any
    merged_list.extend(list1[i:])
    merged_list.extend(list2[j:])

    return merged_list
list1 = [1, 3, 5]
list2 = [2, 4, 6]
result = merge_sorted_lists(list1, list2)
print(result) 

# 5.write a python function that takes a string and returns a dictionary with the frequency count of each character in the string

def char_frequency_count(s):
    frequency = {}
    for char in s:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    return frequency

# Example usage:
input_string = "hello world"
result = char_frequency_count(input_string)
print(result)

# 6.write a python function to convert a roman numerical string into an integer ?
def roman_to_int(s):
    # Define a dictionary for Roman numeral values
    roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    
    total = 0  # Initialize total sum
    prev_value = 0  # Track the previous numeral's value
    
    for char in s:
        value = roman_values[char]
        
        # If the current value is greater than the previous one, adjust the total
        if value > prev_value:
            total += value - 2 * prev_value
        else:
            total += value
        
        # Update the previous value
        prev_value = value
    
    return total

# Example usage:
print(roman_to_int("MCMXCIV")) 


#how to copy lists
l1 = [1,22,3,4]
l2 = l1.copy()
l3 = l1[:]



# 7.write a python function that takes two lists and returns a list of elements that are common to both lists?
def common_elements(list1, list2):
    # Use set intersection to find common elements
    common = list(set(list1) & set(list2))
    return common

# Example usage:
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
print(common_elements(list1, list2))

# 8.write a python function that takes an integer and returns the sum of its digits?
def sum_of_digits(number):
    number = str(abs(number))  # Convert number to string and handle negative values
    return sum(int(digit) for digit in number)  # Sum the integer value of each character

# Example usage:
print(sum_of_digits(1234))  
print(sum_of_digits(-567)) 

f = open("demo.txt",'x')
f.write('new file')
f.close()

f= open("demo.txt")
print(f) # print the file object
f.close() #whenever we open a file we need to close the fie bacuse again we try to open at that time its not open

#Open the file demo.txt using a context manager

with open ('demo.txt') as file:
    x = file.read() # reading the file
print(x)

# open the file in write mode
f = open('demo.txt','w')
f.write('adding the new line,'+'\n this is second line')
f.close()

f = open('demo.txt')
f.readline() #only one line read
f.readline() #only one line read 2nd one
f.close()

f = open('demo.txt', mode = 'r')
print(f.readlines()) #Saperated by comma
f.close()

f = open('demo.txt')
f.read(10) # read upto 10 characters
f.close()

f = open('demo.txt','a')
f.write('\n now the file has more content!')
f.close()

f = open('demo.txt',"r")
f.read() # we can observe the extra line added based on above that

# we can write this way also
f = open('demo.txt',"a+") # a+ mode appends a input first and read after
f.write('im daya\n')
f.seek(0) #adjust the pointer if its 0 it take beggining it its 1 it takes fist character on ward
print(f.read())
f.close()

#adding perticular area


# Step 1: Open the file in read mode to read its contents
with open('demo.txt', 'r') as file:
    content = file.read()

# Step 2: Insert the new content at the specified position
updated_content = content[:10] + ' hello ' + content[10:]

# Step 3: Open the file in write mode to update the file with the modified content
with open('demo.txt', 'w') as file:
    file.write(updated_content)

file.close()

f = open('demo.txt','r')
f.read()
f.close()

fl = open('demo1.txt','x+')
fl.write('this is new file')
fl.close()

import os 
os.remove('demo1.txt')# file removed


fm = open('C:/Users/dayav/OneDrive/Desktop/360digiTMG/study materials@360/EDA/EDA 1/EDA-Datasets/education.csv','r')
print(fm.read())
fm.close()

#Exception handling

# 1. "InvalidInputError:"
#This exception can be used to handle cases where the input provided by the user is invalid. For example:

class invalid(Exception):
    pass
def get_user_age():
    str_age = input('enter the age:')
    try:
        age = int(str_age)
        if age < 0:
            raise invalid('age cannot be negative')
        return age
    except ValueError:
        raise invalid('invalid age u entered')
try:
    age = get_user_age()
    print(age)
except invalid as e:
    print(str(e))

# In Python, a class is a blueprint for creating objects. Classes allow you to define custom data types that encapsulate both data (attributes) and behaviors (methods). Using classes, you can model real-world entities and concepts in your code, making it more organized, reusable, and easier to maintain.

# Key Uses of Classes in Python:
# Encapsulation: Classes bundle data (attributes) and methods (functions) that operate on that data into a single unit. This helps to keep related functionalities together, making the code more modular and organized.

# Reusability: Once a class is defined, you can create multiple objects (instances) of that class, each with its own data. This promotes code reuse, as the same class can be used to create different objects with different states.

# Inheritance: Classes support inheritance, which allows you to create new classes that inherit attributes and methods from existing classes. This helps in reusing and extending existing code without modifying it.

# Polymorphism: Classes in Python support polymorphism, which allows methods in different classes to have the same name but behave differently based on the object that calls them. This is useful when implementing different behaviors in subclasses while maintaining a consistent interface.

# Abstraction: Classes help to abstract complex logic by hiding the internal workings and exposing only the necessary details. This makes the code easier to understand and use.

# write a function that takes a list and a number as input and returns the index of the first occurrence of that number in the list
# What should the function return if the number is not found ?

def find_first_occurence(lst,num):
    try:
        index = lst.index(num)
        return index
    except ValueError:
        # Return -1 if the number is not found in the list
        return -1
# Example usage
lst = [10,20,30,40,50]
num = 30
result = find_first_occurence(lst, num)
print(result)

# Example usage 2 (if not found)
num2 = 60
res_not_found = find_first_occurence(lst, num2)
print(res_not_found)




#######################    Daily practised questions      #######################

import keyword
keyword.kwlist


print(bool(1))


s1 = 'sharart is a trainer'
print('length of s1 is :',len(s1))

name  = "Digitmg"
print(name[0])
print(name[-1])

var = 'hello worldo'
var+"python"

print(var[:6] + 'python')
id(var)
print(var)

var.index('o',var.index("o") + 1)


                                                                 # 22/02/2025
# 1. sum the integer

num = 1111
def sum_all(k):
    sum = 0
    while k>0:
        sum += k%10
        k = k//10
    return sum

sum_all(num)

# or

num = 1111
print(sum(int(digit) for digit in str(num)))

# 2. sorting without builtin function
l = [22,42,35,98,13]

def sort_values(lis):
    sort_lis = []
    while lis:
        min = lis[0]
        for i in lis:
            if i < min:
                min = i
        sort_lis.append(min)
        lis.remove(min)       
    return sort_lis

print(sort_values(l))

# Simple way
l = [22,42,35,98,13]
l.sort()
print(l)

# Simple way using userdefined function
l = [22,42,35,98,13]
def sort_values(lis):
    for i in range(len(lis)):
        for j in range(i+1,len(lis)):
            if lis[i] > lis[j]:
                lis[i],lis[j] = lis[j],lis[i]
    return lis 
print(sort_values(l))

# 3. How do you use list comprehension to create a list of squares for numbers from 1 to 10, 
# excluding even numbers?

k = [i**2  for i in range(10) if i%2 != 0]
print(k)

# 4. Explain the difference between deepcopy() and copy() from the copy module. Provide an example.
'''
The difference between copy() and deepcopy() in Python lies in how they handle nested objects:

copy.copy() creates a shallow copy, meaning it copies the outer object but references the inner objects. Changes to the inner objects affect both the original and copied object.
copy.deepcopy() creates a deep copy, meaning it recursively copies all objects, including nested ones. Changes to the inner objects do not affect the original object.
'''

lis = [[1,2],[3,4]]

import copy # for deep copy
k= copy.deepcopy(lis)
k[1][0] = 43 # [[1, 2], [43, 4]]
lis #[[1, 2], [3, 4]] 

lis = [[1,2],[3,4,5]]
import copy # for shallow copy
sh = copy.copy(lis) # [[1,2],[3,4,5]]
lis[0][1] = 22 
lis ##[[1, 22], [3, 4, 5]]
sh #[[1, 22], [3, 4, 5]]

# 5. Write a Python function that takes a string and returns the count of each character using a dictionary.

def count_chars(str1):
    dict1 = {}
    for j in str1:
        dict1[j] = dict1.get(j, 0) + 1
    return dict1

k = 'dghcvhcvhdc'
print(count_chars(k))
        

# 23/02/2025

print('python' in 'im taken python')
print('p' in [1, 'pyrhon', 'p'])
print('p' in (1, 'p'))
print('p' in {1, 'p'})
print('p' in {'pl': 'p', 1 : 56})

# What is the purpose of the 'is' operator, and how is it different from == ?
a = [1, 2, 3]
b = a # 2563586264960 2563586264960
print(id(a), id(b)) 
c = [1, 2, 3]        # 2563586294208
id(c)

print(a == c)  # True (values are the same)
print(a is c)  # False (different objects in memory)

print(a == b)  # True (values are the same)
print(a is b)  # True (same memory location)

# Conditions and loops

# What is the difference between a for loop and a while loop?

        For Loop	                                                               While Loop
Iterates over a sequence (like a list, range, or string).            	Repeats as long as a condition is True.
Number of iterations is known or fixed.	                                Number of iterations is unknown or depends on a condition.
Uses syntax: for item in sequence:	                                    Uses syntax: while condition:
Stops automatically when the sequence ends.                           	Needs a condition to become False to stop.
More readable for iterating over collections.	                        Suitable for loops that depend on dynamic conditions.

# How do you use the else clause with a loop in Python?
name = 'daya'
i = 0
while i < 10:
    print('hi daya')
    i += 1
else:
    print('loop completed')
    

# How can you break out of nested loops using Python?
for i in range(5):
    for j in range(5):
        if j == 1:
            break
    print(i, j)
    
    
    

# Explain the difference between break, continue, and pass statements.

# break
for i in range(10):
    if i == 5:
        break
    print(i)
    
# Continue
for i in range(10):
    if i == 5:
        continue
    print(i)

# Pass 
for m in range(5):
    pass

    
# How do you iterate over both the index and value of a list using a loop?
fruits = ['apple', 'banana', 'cherry']
for i, j in enumerate(fruits):
    print(i, j)
   
# Conditions (Intermediate Level)

# How do you write a conditional expression using the ternary operator in Python?
'''
A ternary operator is a conditional operator that evaluates a condition and returns one of two values depending on whether the condition is True or False. 
It is called "ternary" because it involves three operands:
Condition
Value if True
Value if False
'''
age = 18
status = "Adult" if age >= 18 else "Minor"
print(status)

# Can you use if statements within list comprehensions? Give an example.
l = [i for i in range(10) if i != 5 ]
l
# What is the difference between using is and == in conditional statements?

x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(x == y)  # True (values are the same)
print(x is y)  # False (different objects in memory)

print(x == z)  # True (same values)
print(x is z)  # True (same object in memory)

# When to Use:
# Use is when checking object identity (e.g., None or Singleton objects):


if x is None:
    print("x is None")
    
# Use == when comparing values:

if x == [1, 2, 3]:
    print("Values match")


# 24/02/2025


# Function with Multiple Arguments:
# Write a function calculate_discount(price, discount_percent) that returns the discounted price. If the discount is greater than 50%, return a message: "Discount too high!". 
# give me the answer


def calculate_dicscount(price, discount_percent):
    discounted_price = price - (price * discount_percent) / 100
    
    if discount_percent > 50:
        return "Discount is too high"
    else:
        return discounted_price

print(calculate_dicscount(price = 200,discount_percent = 55)) 

# Write a recursive function factorial(n) to calculate the factorial of a number

def fact(n):
    if n == 0 or n == 1:
        return 1 
    else:
        return n * fact(n - 1)
    
fact(5)

# Use a lambda function to sort a list of dictionaries by the value of the 'age' key.


people = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}, {'name': 'Charlie', 'age': 35}]

s = sorted(people, key=lambda x: x['age'] )

def sorted_people(st):
    for i in range(len(st)):
        for j in range(i+1, len(st)):
            if st[i]['age'] > st[j]['age']:
                st[i],st[j] = st[j], st[i]
    return st
            
sorted_people(people)

# Function with Variable Arguments:
# Write a function average(*args) that takes any number of arguments and returns their average.

def avg_num(*args):
    return sum(args) / len(args)
avg_num(1, 2, 3, 4, 5)

# Write a function apply_function(func, numbers) that takes a function and a list of numbers as arguments, and returns a list with the function applied to each number.

def apply_function(lst):
    d = []
    for i in lst:
        d.append(i**2)
    return d
l = [1,2,3,4]
apply_function(l)
        

# 25-02-2025
# How do you read and write files using both text and binary modes in Python? Provide examples.

# Reading in text mode
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# Writing in binary mode
with open("example.bin", "wb") as file:
    file.write(b"\x48\x65\x6c\x6c\x6f")  # Writing bytes

# Reading in binary mode
with open("example.bin", "rb") as file:
    content = file.read()
    print(content)

# How can you handle multiple exceptions while reading a file? Provide a code example.
try:
    with open("data.txt", "r") as file:
        content = file.read()
        number = int(content)  # Trying to convert content to an integer
except FileNotFoundError:
    print("Error: The file was not found.")
except ValueError:
    print("Error: Could not convert file content to an integer.")
except (PermissionError, IsADirectoryError) as e:
    print(f"Error: {e}")
else:
    print("File read successfully:", content)
finally:
    print("File handling process complete.")
    
# Python Program: Read File Line by Line with Exception Handling
try:
    # Open the file in read mode
    with open('example.txt', 'r') as file:
        print('File opened successfully. Reading contents....\n')
        
    # Read each line and print it
    for line in file:
        print(line.strip()) # strip() removes trailing newline characters
                
except FileNotFoundError:    
    print('Error : The file was not found.')    
except PermissionError:   
    print('Error : You do not have permission to read this file..')    
except Exception as e:    
    print(f"Error : unexpected error occured: {e}") 
    
else: # executes only if no exception occurs
    print('\n File read successfully without any errors.')
    
finally:
    print('File handling process completed.')
