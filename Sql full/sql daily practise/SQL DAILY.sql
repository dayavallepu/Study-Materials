-- creating the database 
create database timetable;
-- using the database
use timetable;

-- creating a table and varchar means variable and charachter
create table weeklytable(person varchar (20), monday varchar (30), tue varchar(30), total int primary key); -- here primary key is immutable and uniqueness and no null values are accepted

-- dropping the data base that means that data base will be completely deleted we can drop table also but use same database at dropping
drop database first_class;
 
-- altermethod
alter table weeklytable add percentage int; -- here adding one more column but taking this step after 26 



 -- assigning values in the table
 insert into  weeklytable(person,monday,tue,total)values('daya','10','20',30); -- method 1
 insert into weeklytable (person,monday,total)values('ravi','40',40); -- remaining tue will raise null in this case
 
 insert into weeklytable values(person,mon,tue,total),('hemanth','30','40',50); -- method 2 but in this method shuld all colums be prese4nt are else raise error
 
 -- we can add inser multiple rows at a time
 
insert into weeklytable(person,monday,tue,total)values
('nani','40','45',79),
('ashok','3','45',87); 

-- if we want to update or delete any column we do write one code before
set sql_safe_updates =0;

-- update method
update weeklytable set percentage = 80 
where person = 'daya';                 -- method 1 for single perso

update weeklytable set percentage = 80 
where person in ('ravi','hemanth','ashok','nani');  -- method 2 for all members update 80 percentage      (here doubt can we pass differnt percentage in differnt rows)

-- delete method
delete from weeklytable where person is null; 

-- astrick (*) retrives all columns and rows  from the specific table
select * from weeklytable;  -- If you are performing a join between two tables, you can use * to select all columns from both tables.

-- 04/0824 practise...................................................................................................................................................

CREATE DATABASE COLLEGE;
USE COLLEGE;

CREATE TABLE FRIENDS(FIRST_NAME VARCHAR(20) NOT NULL,
NICK_NAME VARCHAR(10),
AGE INT NOT NULL,
LOCATION VARCHAR(20) NOT NULL DEFAULT 'KANIGIRI');
SELECT * FROM FRIENDS;
DESC FRIENDS;
INSERT INTO FRIENDS(FIRST_NAME,NICK_NAME,AGE)
VALUES('DAYAKAR','DAYA',25),('MADHU','MADDY',25),
('JAGADEESH','JAGGU',24);
ALTER TABLE FRIENDS ADD BRANCH VARCHAR(20) NOT NULL;
SET SQL_SAFE_UPDATES = 0;

UPDATE FRIENDS SET BRANCH = 'DS'
WHERE NICK_NAME = 'DAYA';

SELECT * FROM FRIENDS;

DELETE FROM FRIENDS WHERE FIRST_NAME = ('JAGADEEDH');
DELETE FROM FRIENDS WHERE FIRST_NAME = ('DAYAKAR');
DELETE FROM FRIENDS WHERE FIRST_NAME = ('MADHU'); -- HERE SPELLING MISTAKE SO DELETED THE NAMES 
INSERT INTO FRIENDS (FIRST_NAME,NICK_NAME,AGE,BRANCH)
VALUES('DAYAKAR','DAYA',25,'DS'),('MADHU','MADDY',25,'AI');

UPDATE FRIENDS SET BRANCH = 'AI'; -- HERE APPLYING ALL SO NEXT STEP IS MODIFIED TO DAYA BRANCH DS
UPDATE FRIENDS SET BRANCH = 'DS' 
WHERE NICK_NAME = 'DAYA';

UPDATE FRIENDS SET LOCATION = 'GUDUR'
WHERE NICK_NAME = 'DAYA';
DELETE FROM FRIENDS WHERE 
NICK_NAME = 'DAYA';
INSERT INTO FRIENDS (FIRST_NAME,NICK_NAME,AGE,BRANCH)
VALUES('DAYAKAR','DAYA',25,'DS');

-- COMMIT: If all operations are successful and you want to save the changes.
-- ROLLBACK: If there is an error or you decide not to save the changes
START TRANSACTION;
DELETE FROM FRIENDS WHERE 
NICK_NAME = 'DAYA';
COMMIT;
rollback;
SELECT * FROM FRIENDS; 

-- In SQL, BEGIN is often used to mark the start of a transaction.

SELECT * FROM FRIENDS;
--                                    05/08/24
drop database 360digitmg;
DROP DATABASE MYDATABASE;
DROP DATABASE TIMETABLE;
DROP TABLE STUDENTS_DATA;

CREATE TABLE EMPLOYEE(ID SERIAL PRIMARY KEY,
F_NM VARCHAR(20) NOT NULL,
L_NM VARCHAR (20) NOT NULL,
AGE INT NOT NULL,
LOCARION VARCHAR (20) NOT NULL DEFAULT 'HYDERABAD',
DEPT VARCHAR (20) NOT NULL);
SELECT * FROM EMPLOYEE;
ALTER TABLE EMPLOYEE ADD SALARY REAL NOT NULL;
INSERT INTO EMPLOYEE (F_NM,L_NM, AGE,  DEPT,SALARY)
VALUES ('RAVI','KIRAN',25,'BIO TECH',25000.0);
insert into employee (f_nm, l_nm, age, locaRion, dept, salary) values 
('Akhil', 'K', 26, 'Bangalore', 'IT', 42000.00),
('Raja', 'Roy', 35, 'Bangalore', 'IT', 60000.00),
('Shilpa', 'Sharma', 40, 'Chennai', 'IT', 44000.00);

-- DISTINCT = UNIQUE VALUES ,NO REPETETION

SELECT AGE FROM EMPLOYEE;
SELECT DISTINCT AGE FROM EMPLOYEE;
SELECT COUNT(DISTINCT LOCARION ) FROM EMPLOYEE;

-- ORDER BY = SORT THE DATA ,AND ARRANGE DATA IN A SEQUENCE, EITHER IN ASCENDING ORDER (DEFAULT) OR IN DESCENDING ORDER

SELECT F_NM FROM EMPLOYEE; 
SELECT F_NM FROM  EMPLOYEE ORDER BY F_NM;
SELECT F_NM FROM EMPLOYEE ORDER BY AGE;
SELECT F_NM FROM EMPLOYEE ORDER BY f_NM DESC; -- HERE DESCENDING ORDER WILL SHOW 
SELECT * FROM EMPLOYEE ORDER BY AGE ,SALARY; -- THIS IS SECOND LEVEL SORTING INCASDE OF CLASH
SELECT F_NM,L_NM,SALARY FROM EMPLOYEE ORDER BY SALARY;

-- LIMIT = TO PUT A LIMIT ON THE NUMBER OF RECORDS TO BE FETCHED 
-- (FILTER - ELIMINATE WHAT IS NOT REQUIRED)

SELECT * FROM EMPLOYEE LIMIT 3; -- DEFAULT IT WILL TAKE FIRST 3 ROWS
SELECT * FROM EMPLOYEE ORDER BY SALARY LIMIT 3;
SELECT * FROM EMPLOYEE ORDER BY SALARY DESC LIMIT 2;

SELECT ID, F_NM, L_NM  FROM EMPLOYEE ORDER BY ID LIMIT 3 OFFSET 1; -- BEGENING THE SECIND PLACE IT WILL TAKE 3 OCORDS

-- AGGREFATE FUNCTION =SUM,AVG,MIN,MAX, COUNT, COUNT DISTINCT

SELECT COUNT(LOCARION) FROM EMPLOYEE; -- FROM HOW MANY LOCATION FROM PEOPLE JOINING
SELECT COUNT(DISTINCT LOCARION) FROM EMPLOYEE; -- EXACT LOCATION  GIVING
-- TO GIVE AN ALIAS NAME TO THE COLUMN:
SELECT COUNT(DISTINCT LOCARION) AS NUM_OF_LOCATION FROM EMPLOYEE; -- WE GIVE THE COLUMN NAME HERE

-- TO GET THE NUMBER OF PEOPLE 30 YEARS 
SELECT * FROM EMPLOYEE WHERE AGE > 30; -- HERE PRINTED ONLY WHO EVER ABOVE AGE 30 ,ALL MEMBERS SELECTED
SELECT COUNT(f_NM) FROM EMPLOYEE WHERE AGE >35;  -- METHOD 1 COUNT AGE GRETER THAN 35 
SELECT COUNT(*) FROM EMPLOYEE WHERE AGE >35; -- METHOD 2
select count(f_nm) from employee where age>25 and age <35;

-- TOTAL SALARIED BEING PAIN TO EMPLOYEES
SELECT SUM(SALARY) FROM EMPLOYEE; 
SELECT AVG(SALARY) FROM EMPLOYEE;
SELECT MIN(SALARY) FROM EMPLOYEE;
SELECT MAX(SALARY) FROM EMPLOYEE;
-- gives the minimum age, but to know who is the employee:
SELECT F_NM, L_NM ,AGE FROM EMPLOYEE  ORDER BY AGE LIMIT 3;

-- GROUP BY AND HAVING 
SELECT COUNT(LOCARION) FROM EMPLOYEE;
SELECT  COUNT(DISTINCT LOCARION) AS NUM_OF_LOC FROM EMPLOYEE;
select LOCARION,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION; -- GROUP BY DIVIDES GROUPS AND RETRIVE
SELECT LOCARION,DEPT,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION,DEPT;

-- number of employees in each location in each department and above 30 years of age - 
SELECT LOCARION,DEPT ,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION ,DEPT WHERE AGE >30; -- THIS WILL RASISE ERROR

SELECT LOCARION ,DEPT, COUNT(*) FROM EMPLOYEE WHERE AGE >30 GROUP BY LOCARION,DEPT;

-- WHERE MUST BE  USED BEFORE GROUP BY 
-- HERE ,WE CAN USE HAVING AS 'HAVING' WORKS AFTER THE AGGREGATION TO WORK WITH THE AGGREGATED DATA
SELECT LOCARION ,COUNT(*) AS TOTAL FROM EMPLOYEE GROUP BY LOCARION; -- TOTAL IS THE COLUMN NAME CREATED 

-- WE NEED TO LIST OF LOCATIONS WITH MORE THAN 1 EMPLOYEE
SELECT LOCARION ,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION 
HAVING COUNT(*)>1;
-- Number of people from each location-
SELECT LOCARION,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION;

-- Number of people from Hyderabad - 
SELECT LOCARION,COUNT(*) FROM EMPLOYEE GROUP BY LOCARION HAVING LOCARION = 'HYDERABAD';

-- Where is used to filter the records before group by, 
-- and having is after group by to work with aggregated data.

--          ----------------        2ND STUDY MATERIAL                ----------------------


-- Constraints:
-- 5 types of keys supported in MySQL
-- Primary Key,  Foreign Key, Composite Primary Key, Unique Key, Candidate Key
-- Auto Increment
--  DDL and DML commands

-- DDL - Create, Alter, Drop, Truncate
-- DML - Insert, Update, Delete
-- DQL - Select

CREATE DATABASE 360DIGITMG;
USE 360DIGITMG;
CREATE TABLE STUDENTS(FIRST_NAME VARCHAR (20) NOT NULL,
LAST_NAME VARCHAR (20) NOT NULL,
AGE INT NOT NULL,
COURCE VARCHAR (20) NOT NULL DEFAULT ('PDS'));

SELECT * FROM STUDENTS;
INSERT INTO STUDENTS(first_name, last_name, age) VALUES 
('DAYA', 'VALLEPU', 25);
DROP TABLE STUDENTS;
CREATE TABLE STUDENTS(ID INT PRIMARY KEY ,
FIRST_NAME VARCHAR (20) NOT NULL,
LAST_NAME VARCHAR (20) NOT NULL,
AGE INT NOT NULL,
COURCE VARCHAR (20) NOT NULL DEFAULT ('PDS'));
DESC STUDENTS;
insert into studentS(id, first_name, last_name, age, course) values (null, 'Madhavi', 'Kumari', 24, 'DA');
-- this will show an error because Primary Key cannot be null
SELECT * FROM STUDENTS;
insert into studentS(id, first_name, last_name, age)
 values (1, 'Madhavi', 'Kumari', 24);
insert into studentS(id, first_name, last_name, age) values
 (1, 'DAYA', 'VALLEPU', 25);
-- this will show an error that the primary key cannot be duplicated
insert into studentS(id, first_name, last_name, age) values
 (2, 'DAYA', 'VALLEPU', 25);
 
 DROP TABLE STUDENTS;
 
 -- ANOTHER WAY TO CREATE PRIMARY KEY
 CREATE TABLE STUDENT(ID INT, NAME_H VARCHAR(20) NOT NULL,
 AGE INT NOT NULL,COURCE VARCHAR (30) NOT NULL DEFAULT('PDS'),
 PRIMARY KEY(ID));
 DESC STUDENT;
 insert into student(ID,NAME_H, age)
 values (1,'DAYA', 24);
 SELECT * FROM STUDENT;
 DROP TABLE STUDENT;
--                         COMPOSITE PRIMARY KEY

create table sales_rep(
rep_fname varchar(20) not null,
rep_lname varchar(20) not null,
salary int not null
);
select * from sales_rep;

insert into sales_rep(rep_fname, rep_lname, salary) 
values('Anil', 'Sharma', 25000), 
('Ankit', 'Verma', 30000),
('Anil', 'Sharma', 25000);
drop table sales_rep;

create table sales_rep(
rep_fname varchar(20) not null,
rep_lname varchar(20) not null,
salary int not null,
primary key(rep_fname, rep_lname)
);

insert into sales_rep(rep_fname, rep_lname, salary) 
values('Anil', 'Sharma', 25000),
 ('Ankit', 'Verma', 30000),
 ('Anil', 'Sharma', 25000);
--- will throw an error
insert into sales_rep(rep_fname, rep_lname, salary) values
('Anil', 'Sharma', 25000), ('Ankit', 'Verma', 30000), ('Sunil', 'Sharma', 25000);
select * from sales_rep;

-- AUTO-INCREMENT
-- Beginning Auto Increment from a different value (by default it will be 1) 
create table student(id int auto_increment,
first_name varchar(20) not null,
last_name varchar(20) not null, age int not null,
course_enrolled varchar(20) not null default 'Data Analytics',
course_fee int not null, primary key(id));

DESC STUDENT;
insert into student(first_name, last_name, age, course_enrolled, course_fee)
 values ('Sandhya', 'Devi', 28, 'Data Science', 50000), 
 ('Priya', 'Darshini', 25, 'Data Science', 50000); -- ID NOT GIVEN BUT IN AUTO INCREMENT IT WILL NOT RAISE ANY ERROR
 
SELECT * FROM STUDENT;
insert into student(first_name, last_name, age, course_fee) values 
('Ravi', 'Mittal', 28, 30000), ('Akhil', 'K', 25, 30000);
ALTER TABLE STUDENT AUTO_INCREMENT  = 1001; -- HERE WE MENTIONED THE PRIMARY KEY VALUE 1001 ONWARDS
insert into student(first_name, last_name, age, course_fee) VALUE
('DAYA', 'VALLEPU', 24, 30000);
DROP TABLE STUDENT;
DROP TABLE SALES_REP;
-- Primary Key is used to recognize each record in a distinct manner, it will not accept nulls and there can be only one Primary Key in a table.
-- Primary Key could be on multiple columns - Composite Primary Key.


-- UNIQUE KEY - ALLOW ONLY DISTINCT VALUES TO BE ENTERED IN A FIELD.
-- A Table can have multiple Unique Keys. Null entries are allowed.

create table email_registration(
f_name varchar(20) not null,
l_name varchar(20) not null,
email varchar(50) not null
);
insert into email_registration values ('Mohan', 'Bhargav', 'mohan_b@gmail.com');
insert into email_registration values ('Mohan', 'Bhajpai', 'mohan_b@gmail.com');

select * from email_registration;
-- 2 people with the same email id, which should not be allowed
drop table email_registration;
create table email_registration(
f_name varchar(20),
l_name varchar(20),
email varchar(50) unique key,
primary key(f_name,l_name)
);
insert into email_registration values ('Mohan', 'Bhargav', 'mohan_b@gmail.com');
insert into email_registration values ('Mohan', 'Bhajpai', null);
drop table email_registration;

create table email_registration(f_name varchar(20) not null,
l_name varchar(20) not null, email varchar(50) not null unique key,
primary key(f_name, l_name));
desc email_registration;
insert into email_registration values ('Mohan', 'Bhargav', 'mohan_b@gmail.com');
insert into email_registration values ('Mohan', 'Bhajpai', 'mohan_b@gmail.com');
-- second insert statement will throw an error "duplicate entry)
insert into email_registration values ('Mohan', 'Bhajpai', null);
-- won't work as 'null' is given for email, which violates the not null constraint
insert into email_registration values ('Mohan', 'Bhajpai', 'mohan_bhajpai@gmail.com');
insert into email_registration values ('Sakshi', null, 'sakshi@gmail.com');
SELECT * FROM EMAIL_REGISTRATION;
insert into email_registration values ('Sakshi', 'Rajpoot', 'sakshi_r@gmail.com');

-- UNIQUE KEY is used to make sure unique values (no duplicates) are entered into a field.
-- UNIQUE KEY can take NULL also, and we can have multiple unique keys in a table.

-- Difference between Primary Key and Unique Key - 
-- 1) There can be only 1 Primary key, whereas there can be multiple Unique Keys
--  2) Primary Key cannot be NULL, whereas Unique Key could be NULL

-- DDL - Create, Alter, Drop, Truncate 
-- DML - Insert, Update, and Delete
-- DQL - Select

create table student(id int auto_increment,
first_name varchar(20) not null,
last_name varchar(20) not null, age int not null,
course_enrolled varchar(20) not null default 'Data Analytics',
course_fee int not null, primary key(id));

-- Select Statements-
select * from student;   -- it gives all the columns and all the rows/tuples
insert into student(first_name, last_name, age, course_enrolled, course_fee)
 values ('Sandhya', 'Devi', 28, 'Data Science', 50000), 
 ('Priya', 'Darshini', 25, 'Data Science', 50000); -- ID NOT GIVEN BUT IN AUTO INCREMENT IT WILL NOT RAISE ANY ERROR
 
SELECT * FROM STUDENT;
insert into student(first_name, last_name, age, course_fee) values 
('Ravi', 'Mittal', 28, 30000), ('Akhil', 'K', 25, 30000);
ALTER TABLE STUDENT AUTO_INCREMENT  = 1001; -- HERE WE MENTIONED THE PRIMARY KEY VALUE 1001 ONWARDS
insert into student(first_name, last_name, age, course_fee) VALUE
('DAYA', 'VALLEPU', 24, 30000);
USE 360DIGITMG;
SELECT FIRST_NAME,LAST_NAME FROM STUDENT; -- it gives selected columns and all the rows/tuples
SELECT FIRST_NAME,LAST_NAME, COURSE_FEE FROM STUDENT 
WHERE COURSE_FEE < 50000; -- it gives the selected columns and rows meeting the where condition
select first_name, last_name from student
 where binary first_name = 'dAYA'; -- use the binary option to make it case sensitive

SELECT * FROM STUDENT WHERE FIRST_NAME LIKE '____'; -- IN THIS  IM GIVING 4 UNDERSCOE BASED ON THAT 4 CHARCTERS SHOWNED

SELECT * FROM STUDENT WHERE FIRST_NAME LIKE '%AY'; -- give the names which have the character 'a' in the first place

-- UPDATE STATEMENTS

SET SQL_SAFE_UPDATES =0;
UPDATE STUDENT SET COURSE_FEE = 40000 WHERE course_enrolled = 'DATA ANALYST';
SELECT * FROM STUDENT;
UPDATE STUDENT SET COURSE_FEE = COURSE_FEE - 5000;

-- DELETE STATEMENTS
SET SQL_SAFE_UPDATES=0;
DELETE FROM STUDENT WHERE FIRST_NAME  = 'RAVI';


delete from student;    -- deletes all the rows
DROP TABLE STUDENT;
-- DDL - Drop, Alter, Truncate
-- Alter Statement:
alter table student add column location varchar(30) not null default 'Hyderabad';
 alter table student drop column location;
 alter table student modify column first_name varchar(50);
desc email_registration;
ALTER TABLE EMAIL_REGISTRATION DROP PRIMARY KEY;
ALTER TABLE EMAIL_REGISTRATION ADD PRIMARY KEY(F_NAME,L_NAME);
ALTER TABLE EMAIL_REGISTRATION DROP CONSTRAINT EMAIL; -- drop the unique key constraint
ALTER TABLE EMAIL_REGISTRATION ADD CONSTRAINT UNIQUE KEY (EMAIL);
-- Drop - deletes the entire table along with the structure
 -- Truncate - Drops the table and recreates the structure. We can't give a "Where" clause.
-- Delete - Deletes the Rows/Tuples in the table, we can give the "Where" clause and delete exactly what needs to be deleted.


--                  06/8/24

create TABLE PRODUCTS (PRODUCT_ID INT NOT NULL auto_increment,
PRODUCT VARChar (20) NOT NULL,CATEGORY VARCHAR (20) NOT NULL,PRICE INT NOT NULL,PRIMARY key(PRODUCT_ID));
DESC PRODUCTS;
DROP TABLE PRODUCTS;
INSERT INTO PRODUCTS(PRODUCT,CATEGORY,PRICE) VALUES('CARROTS','VEGETABLES',150),
('BANANA','FRUIT',220),('BEANS','VEGETABLES',240),
('APPLES','FRUITS',180),('MANGOS','FRUITS',170),
('KIWI','FRUITS',250),('CUCUMBER','VEGETABLES',110);
SELECT * FROM PRODUCTS;

SELECT PRODUCT FROM PRODUCTS WHERE PRICE <> 150; -- NOT EQUAL TO WE CAN USW != ALSO

SELECT * FROM PRODUCTS WHERE PRICE > 150 OR CATEGORY = 'VEGETABLES'; -- IN THOS CASE ABOVE 150 PRICE ALL PRINTTED AND CATEGORY VEG ALL PRINTED
SELECT * FROM PRODUCTS WHERE PRICE > 150 AND CATEGORY = 'VEGETABLES'; -- IN THIS LINE BOTH CONDITION SHOULD BE TRUE THAT VALUES ONLY PRINTED
SELECT * FROM PRODUCTS WHERE PRICE between 150 AND 200;
SELECT * FROM PRODUCTS WHERE PRICE IN(150,200);
SELECT * FROM PRODUCTS WHERE PRODUCT LIKE 'B%';
SELECT * FROM PRODUCTS WHERE PRODUCT NOT LIKE 'B%';
SELECT PRICE FROM PRODUCTS WHERE PRICE>150;
SELECT * FROM PRODUCTS WHERE PRICE <= ALL (SELECT PRICE FROM PRODUCTS WHERE PRICE>150);
SELECT PRICE FROM PRODUCTS WHERE PRICE>150;
SELECT * FROM PRODUCTS WHERE PRICE <= ANY (SELECT PRICE FROM PRODUCTS WHERE PRICE<150);
SELECT * FROM PRODUCTS WHERE CATEGORY = 'FRUITS' XOR PRICE > 150;


--        6A SQL MATERIAL

USE STUDENTS;

SELECT * FROM EMPLOYEE;
CREATE INDEX IDX_SALARY2 ON EMPLOYEE(SALARY); --  improves the speed of data retrieval operations on a table at the cost of additional storage space and slower writes (inserts, updates, and deletes). Indexes are used to quickly locate 
-- data without having to search every row in a table every time a database table is accessed.
SELECT * FROM EMPLOYEE WHERE SALARY = 60000;
DROP INDEX IDX_SALARY2 ON EMPLOYEE;

Create table product (product_name varchar(20), quantity int);
insert into product values ('Chairs', 20), ('Tables', 5), ('Bookcases', 10), ('Storage', 25);
SELECT * FROM PRODUCT;

SELECT PRODUCT_NAME,QUANTITY,
CASE 
WHEN QUANTITY  > 10 THEN 'MORE THAN 10'
WHEN QUANTITY  < 10 THEN 'LESS THAN 10'
ELSE 'EQUAL TO 10'
END AS QUANTITYTEXT FROM PRODUCT; -- the CASE statement is used to create conditional logic in SQL queries.

USE 360DIGITMG;
DROP TABLE DEPARTMENT; --  a table with a foreign key constraint cannot be dropped directly if other tables reference it.
ALTER TABLE EMP1  DROP CONSTRAINT EMP1_DEPARTMENTID_FKEY; 
DROP TABLE EMP1;
DROP TABLE DEPARTMENT;

CREATE TABLE DEPARTMENT (ID INT PRIMARY KEY,    -- MOTHER TABLE
NAME VARCHAR (10));

insert into department values(1, 'IT'), (2, 'HR');

select * from department;

CREATE TABLE EMP1(ID SERIAL PRIMARY KEY,
NAME VARCHAR (20),SALARY REAL,           --  A REAL DATA TYPE IS USED TO STORE APPROXIMATE NUMERIC VALUE WITH FLOATING POINT PRECISION
DEPARTMENTID INT ,
 foreign key (DEPARTMENTID) REFERENCES DEPARTMENT(ID));
 
-- SERIAL DATATYPE IS USED TO CREATE AUTO INCREMENY COLUMN, TYPICALLY FOR PRIMARY KEYS

insert into emp1 (name, salary, departmentId)
values ('Ravi', 70000, 1), ('Ram', 90000, 1),
 ('Priya', 80000, 2), ('Mohan', 75000, 2), 
 ('Shilpa', 90000, 1);   
 
 insert into emp1 (name, salary, departmentId) values ('Manoj', 80000, 3); -- -- violates the foreign key constraint,CANNOT UPDATE THE CHILD CLASS
 insert into department values(3, 'IT');
 
insert into emp1 (name, salary, departmentId) values ('Manoj', 80000, 3);
 SELECT * FROM EMP1;

-- Q. Find out the names of Employees whose salary is less than the overall average salary?
USE 360DIGITMG;
 SELECT * FROM EMP1;
 SELECT * FROM EMP1 WHERE SALARY < (SELECT AVG(SALARY) FROM EMP1); 

-- Q. Get the highest salary by the department.
SELECT 
    MAX(SALARY)
FROM
    EMP1
GROUP BY DEPARTMENTID;

-- Q. Show the name of the department.

select departmentId, department.name,max(salary) from emp1
 inner join department on emp1.departmentId = department.id
group by departmentId, department.name; 
--  an INNER JOIN is used to combine rows from two or more tables based on a related column between them

-- Q. show the name of the employee also.
select DEPARTMENTID,DEPARTMENT.NAME,EMP1.NAME,MAX(SALARY)
 FROM EMP1 INNER JOIN DEPARTMENT ON EMP1.DEPARTMENTID = DEPARTMENT.ID
 group by DEPARTMENTID,DEPARTMENT.NAME,EMP1.NAME;
  -- This doesn't work as we are now creating groups 
 -- on the combination of Department and Employee.
  
Select department.name, emp1.name, salary 
from emp1 inner join department 
on emp1.departmentId = department.id 
where (departmentId, salary) in 
(select departmentId, max(salary) as salary from emp1 group by departmentId);

-- Q. Selecting the second-highest salary of an employee

SELECT SALARY AS SECOND_HIGH FROM EMP1 ORDER BY SALARY DESC LIMIT 1 OFFSET 1;  

-- Suppose we need those salaries which are less than this:

SELECT SALARY FROM EMP1 WHERE SALARY <( SELECT MAX(SALARY) FROM EMP1);

-- The second maximum means - the maximum of this new list:
SELECT MAX(SALARY) FROM EMP1 WHERE SALARY <(SELECT MAX(SALARY ) FROM EMP1);

--                 6C SQL MATERIAL 

CREATE TABLE COURCES(NAME VARCHAR (20),COURCE VARCHAR(20));
INSERT INTO courceS VALUES('aaa','TABLUE'),('BBB','PYTHON'),('CCC','DATA ANALYTICS'),
('EEE','SQL');
SELECT * FROM COURCES;

CREATE TABLE STUDENTS (NAME VARCHAR (20) ,AGE INT);
INSERT INTO STUDENTS VALUES('AAA',22),('BBB',24),
('CCC',25),('DDD',30);
SELECT * FROM STUDENTS;

-- INNER JOIN
SELECT NAME,COURCE,AGE FROM STUDENTS INNER JOIN COURCE ON NAME = NAME; -- column reference "name" is ambiguous

SELECT STUDENTS.NAME,COURCE, AGE FROM STUDENTS 
INNER JOIN COURCES ON STUDENTS.NAME = COURCES.NAME;

-- LEFT JOIN
SELECT STUDENTS.NAME,COURCE,AGE FROM STUDENTS LEFT JOIN 
COURCES ON STUDENTS.NAME = COURCES.NAME;

-- RIGHT JOIN
SELECT STUDENTS.NAME ,COURCE ,AGE FROM STUDENTS RIGHT JOIN 
COURCES ON STUDENTS.NAME = COURCES.NAME;

-- FULL JOIN NOT SUPPORTED IN MYSQL

--  cross join
select students.name ,age,cources.name ,cource from students cross join cources;

-- left outer join (left only scenario)-
select students.name ,cources.name,age,cource from students left join cources on students.name = cources.name 
where cources.name is null;
-- right outer join
select students.name ,cources.name,age,cource from students left join cources on students.name = cources.name 
where students.name is null;

-- full outer join
-- Full Outer Join: (Not Inner) scenario - 
select students.name, age, cources.name, cource from students full join cources on students.name = cources.name
 where students.name is null or cources.name is null;

-- check constraint
create table school(name varchar (20),schoolname varchar(20)
 default "360digitmg",
age int ,check (age >=18));
select * from school;
desc table school;

insert into school (name,age) values ('priya',17); -- it will through an error check constraint school age violated 
insert into school (name,age) values ('priya',18); 

-- TIME STAMP AND DATE DATATYPES

CREATE TABLE EMP( ID  SERIAL PRIMARY KEY,
NAME VARCHAR(20) NOT NULL,
DEPT VARCHAR (20) NOT NULL,
DATE_OF_JOINING TIMESTAMP NOT NULL DEFAULT NOW(),
SALARY REAL NOT NULL,
LAST_UPDTAED TIMESTAMP NOT NULL DEFAULT NOW());

SELECT * FROM EMP;
 insert into emp (name, dept, salary) 
 values ('Ravi Kiran', 'HR', 40000.00), 
('Priya Darshini', 'IT', 25000.00),
('Mohan Bhargav', 'Finance', 30000.00);
-- Note: MySQL displays DATE values in the 'YYYY-MM-DD' format. WHICH IS IN ISO FORMAT

--                       6D SQL

CREATE VIEW VW_EMPLOYEE AS SELECT
 F_NM,L_NM,SALARY FROM EMPLOYEE WHERE DEPT = 'IT';
SELECT * FROM vw_employee;
-- CREATE VIEQW WITH AVERAGE SALARY
CREATE VIEW VW_AVG AS SELECT 
F_NM,L_NM,SALARY FROM EMPLOYEE WHERE SALARY BETWEEN 40000 AND 50000;
SELECT * FROM vw_avg;
-- Create a view for employees with names starting with 'J'
CREATE VIEW vw_j AS
SELECT F_NM,L_NM,SALARY
FROM employee
WHERE F_NM LIKE 'A%';
SELECT * FROM vw_j;

--              7A SQL
--    UNIONS
CREATE TABLE AUG2016 (DAY INT,CUSTOMER VARCHAR (20),
PURCHASES INT ,TYPE VARCHAR (20));

create table Sept2016 (Day int, Customer varchar(20), 
Purchases real, Type varchar(20));

create table Oct2016 (Day int, Customer varchar(20),
 Purchases int, Type varchar(30));
 
 select * from AUG2016 UNION select * from SEPT2016 UNION select * from OCT2016;
 -- Union does not check for the data type differences.

 INSERT INTO Aug2016(Day, Customer, Purchases, Type)
VALUES
    (1, 'John Doe', 500, 'Mobile'),
    (2, 'Jane Smith', 600, 'Furniture');

INSERT INTO Sept2016(Day, Customer, Purchases, Type)
VALUES
    (1, 'John', 10.20, 'A'),
    (2, 'Smith', 600.30, 'C');

INSERT INTO Oct2016(Day, Customer, Purchases, Type)
VALUES
    (1, 'Sharat', 100, 'ABC'),
    (2, 'Smith', 30, 'D');

select * from Aug2016 UNION select * from Sept2016 UNION select * from Oct2016;

--            UNION COMBINE ALL ABOVE

SELECT STUDENTS.NAME ,AGE ,cource,COURCES.NAME FROM students LEFT JOIN cources
ON STUDENTS.NAME = cources.NAME
UNION
SELECT STUDENTS.NAME ,AGE ,cource,COURCES.NAME FROM students RIGHT JOIN cources
ON STUDENTS.NAME = cources.NAME;

SELECT STUDENTS.NAME ,AGE ,cource,COURCES.NAME FROM students LEFT JOIN cources
ON STUDENTS.NAME = cources.NAME
UNION ALL
SELECT STUDENTS.NAME ,AGE ,cource,COURCES.NAME FROM students RIGHT JOIN cources
ON STUDENTS.NAME = cources.NAME;

--          TRIGGERS
SHOW DATABASES;  -- IT WILL SHOW ALL DATABASES

use 360db;
show tables;
drop table student_info;
create table student_info (stud_id int not null,
STUD_NAME VARCHAR (20) default NULL,
stud_code varchar (20) default null,
subject varchar(20) default null,
marks int default null,
phone varchar(20) default null,
primary key(stud_id));
select * from student_info;

insert into student_info values(101, 01, 'shiv', 'Maths', 50, '966644582');
insert into student_info values(102, 02, 'shivi', 'Maths', 50, '966677582');

CREATE TABLE student_detail (  
  stud_id int NOT NULL,  
  stud_code varchar(15) DEFAULT NULL,  
  stud_name varchar(35) DEFAULT NULL,  
  subject varchar(25) DEFAULT NULL,  
  marks int DEFAULT NULL,  
  phone varchar(15) DEFAULT NULL,  
  Lasinserted Time,  
  PRIMARY KEY (stud_id)
);  


# Next, we will use a CREATE TRIGGER statement to create an after_insert_details trigger on the student_info table.
# This trigger will be fired after an insert operation is performed on the table.
select * from student_detail

delimiter //
create trigger after_insert_detail after insert on student_info for each row
begin
insert into student_detail
values (new.stud_id,new.stud_code,new. stud_name,new.subject,
new.marks,new.phone,curtime());
end //
delimiter ;

select * from student_info;
select * from student_detail;

#we need to add the values then we trigger will work
insert into student_info values(103, 031, 'Amar', 'Datascience', 80, '66644582');
insert into student_info values(104, 032, 'Amer', 'Datascience', 90, '66644583');

##At top we are instering the values then it will load in student_info as well as it will load in the student_detail tables
select * from student_info;
select * from student_detail;

delimiter //
create  trigger before_insert_detail before insert on student_info
for each row
begin
INSERT INTO student_detail 
    VALUES (new.stud_id, new.stud_code,
    new.stud_name, new.subject, new.marks,
    new.phone, CURTIME());
end // 
delimiter ;


insert into student_info 
values(105, 033, 'Aditya', 'Datascience', 90, '66644584');
# This will throw error as the After trigger is active and trying to update the table.

# drop the trigger
drop trigger after_insert_details;

# rerun the insert command
insert into student_info values
(105, 033, 'Aditya', 'Datascience', 90, '66644584');


select * from student_detail;
select * from student_info;

--             8a sql

CREATE TABLE EDUCATION(DATASRNO INT, WORKEX INT,GMAT INT);
SELECT * FROM EDUCATION;

INSERT INTO EDUCATION VALUES(1, 10, 700 );

#2.	import data from CSV file ‘education.csv’

show variables like 'secure_file_priv'; -- HERE SHOWS THE SAFE FILE PATH 
show variables like '%local%' ;
SET OPT_LOCAL_INFILE=1;

LOAD DATA INFILE  "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/education.csv" 
INTO TABLE EDUCATION
FIELDS terminated by ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS ;


SELECT * FROM EDUCATION;

# mean
select avg(workex) as mean_workex from education;

-- median
select workex as median_experience from (select workex,
ROW_NUMBER() over (order by workex)
as row_num, count(*) over () as total_count from education)
as subquery
where row_num = (total_count + 1)/2 or row_num = (total_count +2)/2; 

# MODE
SELECT workex AS mode_workex
FROM (
    SELECT workex, COUNT(*) AS frequency
    FROM education
    GROUP BY workex
    ORDER BY frequency DESC
    LIMIT 1
) AS subquery;

# Second Moment Business Decision/Measures of Dispersion
# Variance
SELECT VARIANCE(workex) AS workex_variance
FROM education;
# Standard Deviation 
SELECT STDDEV(workex) AS workex_stddev
FROM education;

# Range
SELECT MAX(workex) - MIN(workex) AS experience_range
FROM education;

# Third and Fourth Moment Business Decision
-- skewness and kurkosis 

 -- assighnment -2
 create database students;
 use students;
create table sales(sale_id int primary key,
cu_name varchar(20),
product_code char(20),
sale_date date,
sale_time time,
sale_timestamp timestamp,
is_successful boolean,
sale_amount decimal(10,2),
comments text);
select * from sales;
INSERT INTO sales (sale_id, cu_name, product_code, sale_date, sale_time, sale_timestamp, is_successful, sale_amount, comments)
VALUES 
(1, 'daya prince', 'P001', '2023-08-01', '14:30:00', '2023-08-01 14:30:00', TRUE, 250.75, 'VIP customer'),
(2, 'venkatesh', 'P002', '2023-08-02', '15:00:00', '2023-08-02 15:00:00', FALSE, 100.00, 'Customer returned product'),
(3, 'madhu', 'P003', '2023-08-03', '16:15:00', '2023-08-03 16:15:00', TRUE, 99.99, 'Regular customer'),
(4, 'ashok', 'P004', '2023-08-04', '17:45:00', '2023-08-04 17:45:00', TRUE, 300.50, 'Bulk purchase'),
(5, 'jagadeesh', 'P005', '2023-08-05', '10:30:00', '2023-08-05 10:30:00', FALSE, 50.00, 'Discount applied'),
(6, 'vamsi potti', 'P006', '2023-08-06', '11:00:00', '2023-08-06 11:00:00', TRUE, 199.95, 'Customer referral'),
(7, 'hemanth', 'P007', '2023-08-07', '09:45:00', '2023-08-07 09:45:00', TRUE, 120.20, 'Special offer'),
(8, 'mahendra', 'P008', '2023-08-08', '13:15:00', '2023-08-08 13:15:00', TRUE, 180.75, 'Regular customer'),
(9, 'chand basha', 'P009', '2023-08-09', '12:30:00', '2023-08-09 12:30:00', FALSE, 75.50, 'Refund processed'),
(10, 'ravi', 'P010', '2023-08-10', '08:30:00', '2023-08-10 08:30:00', TRUE, 299.99, 'VIP customer');

alter table sales modify column sale_amount float;
alter table sales modify column comments varchar (255);
desc sales; -- this will describe the table sales

-- creating a database of supermart_db
create database supermart_db;
use supermart_db;
create table customers (customer_id char(40) primary key, -- creating a table of customers
customer_name varchar (40),segment varchar(40),age int,
country varchar(40),region varchar (40),city varchar(30),state varchar(40),postal_code char(40) );
select* from customers;

load data infile"C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Customer.csv" -- adding the csv file
into table customers 
fields terminated by ","
enclosed by'"'
lines terminated by'\r\n'
ignore 1 rows;

-- crating a another table products
create table products(product_id char(30),
category varchar (30),sub_category varchar(30),
product_name varchar(50)); 
select * from products;
load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Product.csv"  -- loading th edata file into the products
into table products
fields terminated by ','
enclosed by '"'
lines terminated by '\r\n'
ignore 1 rows;
ALTER TABLE products MODIFY product_name VARCHAR(255); -- here im increased the column length

-- creating a new table sales in the supermart_db databse
create table sales (order_line int ,order_id char(30),
order_date varchar(30),ship_date varchar(30),ship_mode varchar(330),customer_id char(30),
product_id char(30),sales decimal(15,3),quantity int,discount float, profit decimal(10,4));

select * from sales;
load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Sales.csv"
into table sales
fields terminated by ','
enclosed by'"'
lines terminated by'\r\n'
ignore 1 rows ;
drop table sales;
-- SELECTION OPERATORS:- (FILTERING):- in, like, between

-- 2a.a.	Get the list of all the cities where the region is north or east without any duplicates using the IN statement.

select distinct city  from customers where region in ('north','south');  -- distinct keyword ensures there is no duplicate entries


-- b.	Get the list of all orders where the ‘sales’ value is between 100 and 500 using the BETWEEN operator

select * from sales where sales between 100 and 500;

-- c.	Get the list of customers whose last name contains only 4 characters using LIKE.
select * from customers  where customer_name like '____';-- but there is no las name column in the file

-- SELECTION OPERATORS:- ordering
-- 1.	Retrieve all orders where the ‘discount’ value is greater than zero ordered in descending order basis ‘discount’ value

select * from sales  where  discount > 0 order by discount desc;

-- 2.	Limit the number of results in the above query to the top 10.

select * from sales  where  discount > 0 order by discount desc limit 10 ; 

-- Aggregate operators:-
-- 1.	Find the sum of all ‘sales’ values.
select sum(sales) as total_sales from sales; -- aceccing the sum of sales values to total_sales column
-- 2.	Find count of the number of customers in the north region with ages between 20 and 30
select count(*) as cust_count from customers where region = 'central' and age between 20 and 30; 
-- 3.	Find the average age of east region customers
select avg(age) as avg_age from customers where region = 'east';

-- 4.	Find the minimum and maximum aged customers from Philadelphia
SELECT MIN(AGE) AS MIN_AGE,MAX(AGE) AS MAX_AGE FROM CUSTOMERS WHERE CITY = 'Philadelphia';

-- GROUP BY OPERATORS:-
-- 1.	Create a display with the information below for each product ID.
-- a.	Total sales (in $) order by this column in descending 
SELECT PRODUCT_ID,SUM(SALES) AS TOTAL_SALES  FROM  SALES
GROUP BY PRODUCT_ID ORDER BY TOTAL_SALES DESC;
-- b.	Total sales quantity
SELECT PRODUCT_ID, SUM(QUANTITY) AS SALES_QUANTITY FROM SALES 
GROUP BY PRODUCT_ID ORDER BY SALES_QUANTITY;

-- C. THE NUMBER OF ORDERS
SELECT PRODUCT_ID ,COUNT(ORDER_ID) AS NUM_ORDERS FROM SALES
GROUP BY PRODUCT_ID ORDER BY NUM_ORDERS;
-- d.	Max Sales value
SELECT PRODUCT_ID ,MAX(SALES) AS MAX_SALES FROM SALES 
GROUP BY PRODUCT_ID ORDER BY MAX_SALES;
-- e.	Min Sales value
SELECT PRODUCT_ID ,MIN(SALES) AS MIN_SALES FROM SALES 
GROUP BY PRODUCT_ID ORDER BY MIN_SALES;

-- f.	Average sales value
SELECT PRODUCT_ID ,AVG(SALES) AS AVG_SALES FROM SALES 
GROUP BY PRODUCT_ID ORDER BY AVG_SALES;
-- 2.	Get the list of product ID’s where the quantity of product sold is greater than 10
SELECT PRODUCT_ID FROM SALES GROUP BY PRODUCT_ID  HAVING SUM(QUANTITY)>10;

-- Joins
-- 1.	Run the below query to create the datasets.
-- a.	/*retrieve sales table from the supermart_db (sales dataset contains multiple years data)*/
create  database supermart_db;
use supermart_db;
create table sales (order_line int ,order_id char(30),
order_date varchar(30),ship_date varchar(30),ship_mode varchar(330),customer_id char(30),
product_id char(30),sales decimal(15,3),quantity int,discount float, profit decimal(10,4));
select * from sales;
 
-- b.	/* Counting the number of distinct customer_id values in sales table */
-- distinct keyword in sql is used to remove duplicate values from the result of query
select count(distinct customer_id) as dist_cust_values from sales;

-- c.	/* Customers with ages between 20 and 60 */
-- ●	create table customer_20_60 as select * from customers where ages between 20 and 60;
create table customers_20_60 as select * from customers where age between 20 and 60; 
select * from customers_20_60; -- retreving all values 

-- ●	select count (*) from customer_20_60;
select count(*) from customers_20_60;

-- 2.	Find the total sales that are done in every state for customer_20_60 and the sales table
-- Hint: Use Joins and Group By comman

SELECT 
   c.state,
    SUM(CAST(s.sales AS DECIMAL(10, 2))) AS total_sales
FROM 
    customers_20_60 c
JOIN 
    sales s ON c.customer_id = s.customer_id  -- Explicitly referencing the table alias
GROUP BY 
    c.state;

-- 3.	Get data containing Product_id, Product name, category, total sales value of that product, and total quantity sold. (Use sales and product tables)
select p.Product_id, p.Product_name , p.category, sum(s.sales) as Total_sales,sum(s.quantity) as Total_qty
from products p
join sales s on p.product_id = s.product_id
group by p.Product_id, p.Product_name , p.category; 

--                                      Assignmrnt -7
-- a)	Create a database named student_db.
create database student_db;
use student_db;

-- b)	Create a table named students_details with columns id (integer), name (varchar), age (integer), and grade (float). id should be set as the primary key.

create table student_details 
(stud_id int,name varchar(20),stud_age int,stud_grade FLOAT,primary key(stud_id));

select * from student_details;
-- c)	Insert any four records into students_details
insert into student_details (stud_id,name,stud_age,stud_grade) values
(1,'daya',25,99.8),(2,'madhu',24,87.6),(3,'jagadeesh',23,57.6),(4,'ashok',26,78.4);

-- d)	Create a new table named students_details_copy with the same columns as students_details. id should also be set as the primary key.
create table student_details_copy 
(stud_id int,stud_name varchar(20),stud_age int,stud_grade FLOAT,primary key(stud_id));
select * from student_details_copy;
-- e)	Create a trigger named after_insert_details that inserts a new record into students_details_copy every time a record is inserted into students_details.
delimiter //
create trigger  after_insert_details
after insert on student_details
for each row
begin
insert into student_details_copy
values(new.stud_id,new.name,new.stud_age,new.stud_grade);
end // 
delimiter ;
-- f)	Insert a new record into students_details.
insert into student_details (stud_id,name,stud_age,stud_grade) values (5,'poorni',24,99.7);
-- g)	check whether a record is filling in students_details_copy as you insert value in students_details.


SELECT * FROM STUDENT_DETAILS;
select * from STUDENT_DETAILS_COPY; 
-- 2)	Write an SQL question that accomplishes the following tasks:
-- a)	use student_db , 
use dtudent_db;

delimiter //
create trigger update_grade
before update on student_details
for each row
begin
if new.stud_age < 18 then

-- c)	If the updated record has an age value less than 18, multiply the grade by 0.9.

set new.stud_grade = new.stud_grade * 0.9;
elseif new.stud_age between 18 and 20 then
set new.stud_grade = new.Stud_grade *1.1;

-- d)If the updated record has an age value between 18 and 20 (inclusive), multiply the grade by 1.1.
else 
-- e)	If the updated record has an age value greater than 20, multiply the grade by 1.05.
set new.stud_grade = new.stud_grade * 1.05;
end if;
end//
delimiter ;

-- f)	Update the age value of one of the records in students_new to see the trigger in action.

set sql_safe_updates = 0; -- here we need to update the values safely
update student_details set stud_age =19
where stud_id = 1;
select * from student_details where stud_id = 1;

-- 4)	What is the purpose of the INSTEAD OF DELETE trigger operator in SQL? in SQL.
-- Purpose of the INSTEAD OF DELETE Trigger:
-- Handling Deletes on Views:

-- Complex Views: In many database systems, views can be complex, involving multiple tables or calculated columns. Directly deleting data from such views might not be allowed or may not behave as expected. The INSTEAD OF DELETE trigger allows you to define how deletions should be handled for the underlying tables when a DELETE operation is attempted on the view.
-- Virtual Deletions: In some cases, you might not want to delete data physically from a table but instead want to mark it as deleted (e.g., setting a deleted flag). The INSTEAD OF DELETE trigger can implement this behavior by updating the relevant columns rather than deleting the record.
-- Custom Deletion Logic:

-- Cascading Deletes: While foreign keys with ON DELETE CASCADE can handle simple cascading deletes, there are cases where more complex logic is needed. For example, you might need to perform additional checks or update related tables in non-standard ways. The INSTEAD OF DELETE trigger allows you to execute this custom logic before the deletion process.
Conditional Deletes: You can use the INSTEAD OF DELETE trigger to conditionally allow or deny deletions based on specific criteria (e.g., preventing deletion of records that are still in use or have certain dependencies).
Maintaining Data Integrity:

Soft Deletes: Instead of permanently removing records from the database, the INSTEAD OF DELETE trigger can be used to perform a "soft delete," where a record is marked as deleted (e.g., by setting a deleted column to TRUE) without actually removing it from the database. This helps maintain data integrity and allows for easy recovery of deleted records if needed.
Example of an INSTEAD OF DELETE Trigger:
Suppose you have a view active_users_view that shows only active users from a users table, and you want to prevent actual deletion from the table but instead mark the user as inactive.

sql
Copy code
CREATE TRIGGER instead_of_delete_user
INSTEAD OF DELETE ON active_users_view
FOR EACH ROW
BEGIN
    -- Instead of deleting the user, set the 'is_active' column to 0 (inactive)
    UPDATE users
    SET is_active = 0
    WHERE user_id = OLD.user_id;
END;
-- How It Works:
-- INSTEAD OF DELETE ON active_users_view: This trigger is fired when a DELETE operation is attempted on the active_users_view.
-- Custom Logic: Instead of deleting the record, the trigger updates the is_active column in the users table, marking the user as inactive.

-- 1.write an Sql query to select all columns from a table named “students” where the age is greater than 18.
use  students;
select * from students where age >18;
-- 2.how would you write an SQL query to find the avg salary from a table named ‘employees’?
select avg(salary) as avg_salary from  employee;

-- write an  SQL query to join two tables ‘orders’ and ‘customers_id’ and select the ‘order_id’ and ‘order_date’ and ‘customer_name’
select order_id,order_date,customer_name from 
orders o 
join customer c
on 
o.customers_id =c. customers_id;

-- how would  you write an sql query to select the names of all employees from the employees table who where hired after January 1st 2023?
use 360digitmg;
select name from emp where date_of_joining > '2023-01-01';

-- write an SQL query to update the ‘price’ of a product in the ‘products’ table where the product_id is 101 to a new value 29.99
use students;
set sql_safe_updates = 0;
update products set  price = 29.99 where product_id = 101;
select * from products;

-- 1.write an sql query to find the total sales for each product category from a table named ‘sales’ with columns ‘category’ and ‘amount’?
use supermart_db;
select p.category,sum(s.sales) as Total_sales from products p
join sales s on 
p.product_id = s.product_id
group by p.category;

-- 

set sql_safe_updates = 0;
use 360digitmg;
create table superstore_data(Row_ID int,order_id char(30),Order_date varchar (20),Ship_date varchar(30),Ship_mode varchar(20),customer_id varchar(20),Customer_name varchar (100),
segment varchar(20),country varchar(20),city varchar(20),state varchar(20),postal_code int,
region varchar(20),product_id char (20),category varchar(20),sub_category varchar (20),product_name varchar(500),Sales int,Quantity int,Discount float,profit float);
load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/Superstore_dataset.csv"
into table superstore_data 
fields terminated by ','
enclosed by'"'
lines terminated by'\r\n'
ignore 1 rows ;
drop table superstore_dataset;


----------------------------------- 01/10/24 prectise daily  -------------------------------
create database daily_practise;
show databases;
use  ds;
show tables;
desc anime;
select * from anime;

create database daily_practise;
use  daily_practise;

CREATE TABLE EMPLOYEE(
FIRST_NAME VARCHAR (30) not null,
LAST_name varchar(20) not null,
age int not null,
salary int not null,
location varchar(20) not null default "Banglore");

select * from employee;
desc employee;
insert into employee values ('daya','vallepu',25,90000,'gudur');
insert into employee values ('ravi','vallepu',26,50000); -- through error because this step should write all the columns 

-- instead of above step we can use 
insert into employee (first_name, last_name,age,salary) values ('ravi','vallepu',26,50000); 
select * from employee;

drop table employee; # dropped the table

















