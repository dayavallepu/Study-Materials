"""
Introduction to Object-Oriented Programming (OOP) in Python

Welcome to the Object-Oriented Programming (OOP) module of our Python course!
Whether you're new to programming or looking to deepen your understanding,
mastering OOP is essential for building robust, scalable, and maintainable software.

This script covers the following OOP concepts:
1. Classes and Objects
2. Encapsulation
3. Inheritance
4. Polymorphism

Additionally, it includes examples to illustrate each concept.
"""

# ----------------------------------------
# 1. Classes and Objects
# ----------------------------------------

'''A class is a blueprint or a template for creating objects. 
It defines a set of attributes (variables) and methods (functions) 
that describe the behavior of the object.

An object is an instance of a class. When a class is defined, no memory is allocated until an object of that class is created. 

Objects have:

State: Represented by attributes (variables)
Behavior: Represented by methods (functions)

Class - A blueprint/template for creating objects.
Object - An instance of a class with specific values.
Constructor (__init__) - A special method that initializes an object’s attributes.
Attributes - Variables hold data.
Methods - Functions inside a class that define behaviors.
self keyword - Represents the instance of the class, allowing access to attributes and methods.
'''

class Car: # class keyword is used to create class called 'Car'
    def __init__(self, make, model, year):  # Without self, Python wouldn't know where to store the value!
        '''# __inti__ is constructor. This is called automatically when a new object of the class
        is created. A special method that initializes an object’s attributes. self refers to the 
        instance of the class. make, model, and year are parameters passed when creating a Car object.'''
    
        self.make = make # make, model, year are attributes
        self.model = model
        self.year = year # self allows us to attach the values to the object being created.
    
    def display_info(self): # method names display_info is created inside class Car
        print(f"{self.year} {self.make} {self.model}") # fetch values from object's attributes

my_car = Car("Toyota", "Corolla", 2020) # Creating an instance (object) my_car of Car class
my_car.display_info()  # This calls the display_info method on the my_car object.



# ----------------------------------------
# 2. Encapsulation
''' It refers to restricting direct access to some attributes (variables) and 
methods (functions) of an object, to prevent accidental modification of data.

Encapsulation hides the internal state of an object and allows access to it only 
through controlled methods (like deposit() and get_balance() in this example).
This is done using private attributes 
(indicated by double underscores __ before the variable name).
It ensures data security and prevents direct modification of sensitive information.'''

# ----------------------------------------

class BankAccount: # Defines a class called 'BankAccount'
    def __init__(self, account_holder, balance = 0): # The __init__ method is a constructor. It runs automatically when a new object is created.
        self.account_holder = account_holder  # Public attribute    
        self.__balance = balance  # Private attribute (cannot be accessed directly)
    
    def deposit(self, amount): # 'deposit' method is created
        if amount > 0:
            self.__balance += amount
            print(f"Deposited: {amount}")
        else:
            print("Deposit amount must be positive.")
    
    def get_balance(self): # 'get_balance' method is created. It provides controlled access to the private attribute __balance.
        return self.__balance

# Using the BankAccount class
account = BankAccount("Rajnikanth", 100) # 'account' is an object (instance) of the 'BankAccount' class.
account.deposit(100)

# Accessing public attribute
print(account.account_holder)

# Accessting private attribute
print(account.__balance)  # Since __balance is private, it cannot be accessed directly from outside the class.

# Accessing private attribute via a method 'get_balance'
print(f"Current Balance: {account.get_balance()}") 


# ----------------------------------------
# 3. Inheritance 
# ----------------------------------------

# Parent Class
class Vehicle:
    def __init__(self, make, model):
        self.make = make # Attributes are variables inside a class that store object data.
        self.model = model

    def start_engine(self): # Methods are functions inside a class that define what an object can do.
        print("Engine started.")

# Child Class (Inherits from Vehicle)
class Motorcycle(Vehicle): 
    def wheelie(self):
        print("Doing a wheelie!")

#  Creating an instance of Motorcycle
bike = Motorcycle("Harley-Davidson", "Sportster")

# Using inherited method
bike.start_engine()  # Output: Engine started.

# Using child class method
bike.wheelie()  # Output: Doing a wheelie!

# ----------------------------------------
# 4. Polymorphism, means different classes can have the same method name but behave differently.
# ----------------------------------------
'''self refers to the object itself.
It allows each object to have its own data and behavior.
Without self, Python wouldn’t know which object’s data to use.'''

# Class for Dog. A class is a blueprint or template for creating objects.
class Dog:
    def make_sound(self):
        print("Woof!")  

# Class for Cat
class Cat:
    def make_sound(self):
        print("Meow!")  

# A function that takes any animal and calls make_sound().
def animal_sound(animal):
    animal.make_sound()  # Calls the correct method

# Creating Dog and Cat objects
dog = Dog() # An object is a specific instance of a class.
cat = Cat()

# Calling function with different animals
animal_sound(dog)  # Output: Woof!
animal_sound(cat)  # Output: Meow!


# ----------------------------------------
# 5. Abstraction means hiding the details and showing only the important parts.
# You use a TV remote but don’t need to know how the internal circuits work!
# In Python, abstraction hides complex logic and provides a simple interface.
# ----------------------------------------

# Abstraction Hides implementation details, shows only important methods.
# Abstract Class (Shape) Cannot create objects, only acts as a blueprint.
# Abstract Methods (area, perimeter) Must be implemented by child classes.
# Child Classes (Rectangle, Circle)	Define their own versions of area() and perimeter().

from abc import ABC, abstractmethod 
# ABC → Stands for Abstract Base Class, which means this class cannot be used to create objects.
# It prevents creating objects from the class.
# It forces child classes to implement certain methods.
# Think of it as a blueprint – You cannot use a blueprint directly; 
# you must build something based on it.

# abstractmethod → Forces child classes to define specific methods.
# @abstractmethod means the method must be defined in child classes.
# It does nothing in the parent class (pass keyword).
# It must be implemented by any class that inherits from it.
# Think of it as a contract – If a child class inherits Shape, it must define area() and perimeter().

class Shape(ABC):  # Abstract class. Shape is an abstract class (a blueprint for shapes). Cannot be used directly. 
    @abstractmethod
    def area(self): # area() is defined but not implemented. Any child class MUST define its own area() method.
        pass  # area() and perimeter() must be defined in child classes.
    
    @abstractmethod
    def perimeter(self): # perimeter() is also enforced but not implemented.
        pass  # Must be implemented by child classes

#  What Happens If a Child Class Doesn't Implement These Methods?
# ERROR! Because area() and perimeter() is missing.
# Shape(ABC) is like a TV remote without buttons – it’s incomplete!
# Child classes (Rectangle, Circle) add the buttons (methods).

class Rectangle(Shape):  # Child class # Rectangle inherits from Shape and implements area and perimeter methods.
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):  # Implements area method
        return self.width * self.height
    
    def perimeter(self):  # Implements perimeter method
        return 2 * (self.width + self.height)

class Circle(Shape): # Circle also inherits from Shape and implements its own formulas.
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * (self.radius ** 2)
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Attempting to instantiate Shape will raise an error
shape = Shape()  # Raises TypeError # You cannot create an object of Shape directly.

# Creating instances of Rectangle and Circle
rect = Rectangle(5, 3)
circle = Circle(4)

# We only call area() and perimeter() without worrying about how they work inside!
# This is abstraction → Hiding complex formulas and exposing a simple interface!
print("Rectangle Area:", rect.area())          # Output: Rectangle Area: 15
print("Rectangle Perimeter:", rect.perimeter())# Output: Rectangle Perimeter: 16
print("Circle Area:", circle.area())            # Output: Circle Area: 50.26544
print("Circle Perimeter:", circle.perimeter())  # Output: Circle Perimeter: 25.13272

# ----------------------------------------
# 6. Attributes store information about an object.
# Types of Attributes in This Example:
'''Public (name) – Accessible from anywhere.
Protected (_position) – Should only be accessed inside the class or 
subclasses (not enforced, just convention).
Private (__salary) – Cannot be accessed directly outside the class.
Class Attribute (company) – Shared by all objects of the class.'''

# ----------------------------------------

class Employee:
    # Class attribute
    company = "AiSPRY"  # Shared by all instances Class. Attribute – Belongs to the class, not just one object.
    
    def __init__(self, name, position, salary): # Constructor (__init__) – Runs when an object is created.
        self.name = name              # Public attribute
        self._position = position     # Protected attribute
        self.__salary = salary        # Private attribute
    
    def get_salary(self): # We use get_salary() instead of accessing it directly.
        return self.__salary
    
    def set_salary(self, amount):
        if amount > 0:
            self.__salary = amount
        else:
            print("Invalid salary amount.")

# Creating an instance of Employee
emp = Employee("Rajinikanth", "Developer", 70000)

# Accessing public attribute
print(f"Name: {emp.name}")  # Works because public attributes can be accessed directly.

# Accessing protected attribute (conventionally should not be accessed directly)
print(f"Position: {emp._position}")  # Works, but it's not recommended (only a convention, not enforced).

# Attempting to access private attribute directly will raise an AttributeError
print(f"Salary: {emp.__salary}")  # Raises AttributeError

# Using getter method to access private attribute. 
print(f"Salary: {emp.get_salary()}")  # Works fine because get_salary() gives controlled access.

# Using setter method to modify private attribute
emp.set_salary(75000)
print(f"Updated Salary: {emp.get_salary()}")  # Output: Updated Salary: 75000

# Accessing class attribute
print(f"Company: {Employee.company}")  #  Class attributes belong to the class, not just one object.

# Modifying class attribute
Employee.company = "360DigiTMG"
print(f"Updated Company: {emp.company}")  # Output: Updated Company: AiSPRY

# ----------------------------------------
# 7. Method is just a function inside a class that does something with an object.
# ----------------------------------------

class MathOperations: # Defines a class called MathOperations. This class will have methods to perform math calculations.
    # Class attribute
    operation_count = 0 # operation_count belongs to the entire class, not just one object. It keeps track of how many additions were done.    
    def __init__(self): # Constructor runs when a new object is created. Here, it does nothing (pass) because we don't need any setup.
        pass  
    
    def add(self, a, b): # add() is an instance method (belongs to the object). self means this method can use object attributes.
        MathOperations.operation_count += 1 
        return a + b 
    
    @classmethod # Belongs to the class, not an object.
    def get_operation_count(cls): # Uses cls instead of self to work with the class.
        return cls.operation_count # Returns how many times add() was used. 
    
    @staticmethod # Does not use self or cls because it doesn't need object/class data.
    def multiply(a, b): # Works like a normal function, just inside a class.
        return a * b

# Creating an instance of MathOperations
math_ops = MathOperations() # Creates an object math_ops of MathOperations.x

# Using instance method
sum_result = math_ops.add(10, 5)
print(f"Sum: {sum_result}")  # Output: Sum: 15

# Using class method
print(f"Operations Count: {MathOperations.get_operation_count()}")  # Output: Operations Count: 1

# Using static method
product = MathOperations.multiply(4, 3)
print(f"Product: {product}")  # Output: Product: 12

# ----------------------------------------
# Inheritance and Method Overriding Example
'''If a child class defines a method with the same name as in the parent class, 
it overrides the parent’s method. This means the child class changes the behavior of the method.'''
# ----------------------------------------

class Animal: # Defines a class called Animal. This is the Parent Class (also called Super Class).
    def __init__(self, name): # Constructor Method (__init__). This method runs when an object is created.
        self.name = name # Assigns the value of name to the object's attribute (self.name). Now, every animal will have a name when created.
    def speak(self): # speak() is a method that prints a message. This will be overridden by child classes (Dog and Cat).
        print(f"{self.name} makes a sound.")

class Dog(Animal): # Dog inherits from Animal. This means Dog gets all methods and attributes from Animal.
    def speak(self): # Overrides speak() method.
        print(f"{self.name} says: Woof!") # Instead of "makes a sound", a Dog now barks ("Woof!").

class Cat(Animal):
    def speak(self): # Overrides speak() method.
        print(f"{self.name} says: Meow!")

# Creating instances of Dog and Cat
dog = Dog("Buddy") # Creates an object dog of class Dog with name "Buddy".
cat = Cat("Whiskers") # Creates an object cat of class Cat with name "Whiskers".

dog.speak()  # Calls speak() on dog. Since Dog overrides the speak() method, it prints: "Buddy says: Woof!".
cat.speak()  # Calls speak() on cat. Since Cat overrides the speak() method, it prints: "Whiskers says: Meow!".

# END OF MODULE