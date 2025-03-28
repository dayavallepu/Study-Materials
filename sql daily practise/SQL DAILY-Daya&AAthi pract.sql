create database practise;
use practise;
create table employee(first_name varchar(30) not null,
last_name varchar(30) not null,
age int not null,
salary int not null,
location varchar(20) not null default "Andhra Pradesh");
select * from employee;
desc employee;
insert into employee values("daya","vallepu",24,1000000,"Andhra");
drop table employee;


create table education(datasrno int,workex int,gmat int);
load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/education.csv"
into table education
fields terminated by ","
enclosed by'"'
lines terminated by "\r\n"
ignore 1 rows;

select * from education;
drop table education;

create table employee(id serial primary key,
f_nm varchar(20) not null,
l_nm varchar(20) not null,
age int not null,
location varchar(20) not null default "Andhra pradesh",
dept varchar(20) not null);
insert into employee(f_nm ,
                     l_nm ,
                     age,
                     dept) values("ravi","kiran",25,"IT");

set sql_safe_updates = 0;
use ele_price_forecast;
alter table employee add column salary int not null;
select * from employee;
update employee  set salary = 80000
where f_nm = "ravi";

insert into employee (f_nm, l_nm, age, dept, salary) values 
('Priya', 'Darshini', 28, 'HR', 32000.00),
('Mohan', 'Bhargav', 35, 'IT', 40000.00),
('Manoj', 'Bajpai', 40, 'IT', 45000.00);

insert into employee (f_nm, l_nm, age, location, dept, salary) values 
('Akhil', 'K', 26, 'Bangalore', 'IT', 42000.00),
('Raja', 'Roy', 35, 'Bangalore', 'IT', 60000.00),
('Shilpa', 'Sharma', 40, 'Chennai', 'IT', 44000.00);

select location from employee;
select distinct location from employee;
select count(distinct location)  as location_count from employee;
use practise;
select f_nm from employee;

select f_nm from employee order by  f_nm;
select f_nm from employee order by  f_nm desc;
select f_nm, age  from employee order by age;
select * from employee order by age,salary;

select * from employee  order by age limit 3 ;
select * from employee order by age ,salary limit 2 offset 3;

select count(*) from employee;
select count(distinct location) from employee;
select f_nm from employee where age > 30;
select sum(salary) from employee;

select count(location) from employee;
select count(distinct location) from employee;
select location, count(*) from employee group by location;

select location, dept, count(*) from employee group by location,dept;
select location, dept,count(*)  from employee where age >20 group by location,dept ; 


use practise;
create table classwork(name varchar(30),age int not null,cource varchar(30));
desc classwork; -- describing the data
insert classwork values("daya",25,"PDS");
select * from classwork; -- retriving the data
drop table classwork;









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
SELECT COUNT(DISTINCT LOCARION) FROM EMPLOYEE; -- EXACT LOCATION COUNT GIVING
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
Purpose of the INSTEAD OF DELETE Trigger
Handling Deletes on Views:

Complex Views: In many database systems, views can be complex, involving multiple tables or calculated columns. Directly deleting data from such views might not be allowed or may not behave as expected. The INSTEAD OF DELETE trigger allows you to define how deletions should be handled for the underlying tables when a DELETE operation is attempted on the view.
Virtual Deletions: In some cases, you might not want to delete data physically from a table but instead want to mark it as deleted (e.g., setting a deleted flag). The INSTEAD OF DELETE trigger can implement this behavior by updating the relevant columns rather than deleting the record.
Custom Deletion Logic:

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

use  ds;


--                                            07/10/24 30Min practise
create table education (datasrno int,workex int, gmat int);
select * from education;
desc education;
insert education values (1,40,400);
-- importing data drom csv file 'education.csv'
show variables like 'secure_file_priv';
show variables like '%local%';

load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/education.csv" into table education
fields terminated by ','
enclosed by '"'
lines terminated by '\r\n'
ignore 1 rows;

select * from education;
delete from education where datasrno  = 1 and workex = 40;

-- ----------- 3b 
-- Clauses:
-- Distinct, Order By, Limit, Aggregate functions, Group By, Having
use practise;
drop table employee;
create table employee (id serial primary key,
f_nm varchar(20) not null,
l_nm varchar(20) not null,
age int not null,
location varchar(20) not null default 'Banglore',
dept varchar(20) not null);
desc employee;
alter table employee add salary real not null;

select * from employee;

insert into employee (f_nm,l_nm,age,dept,salary) values ('daya', 'vallepu', 25, 'ds', 500000);
insert into employee (f_nm,l_nm,age,dept,salary) values ('Priya', 'Darshini', 28, 'HR', 32000.00),
('Mohan', 'Bhargav', 35, 'IT', 40000.00),
('Manoj', 'Bajpai', 40, 'IT', 45000.00);
 insert into employee (f_nm, l_nm, age, dept, salary) values 
('Priya', 'Darshini', 28, 'HR', 32000.00),
('Mohan', 'Bhargav', 35, 'IT', 40000.00),
('Manoj', 'Bajpai', 40, 'IT', 45000.00);
-- Distinct - Unique values, no repetetion
select location from employee;
select distinct location from employee;
select distinct dept from employee;
select count(distinct dept)  from employee;

-- Order by - sort the data , and arrange the data in a sequence, either in ascending order (default) or descending order (desc)

select f_nm from employee;
select f_nm from employee order by f_nm;
select f_nm from employee order by f_nm desc;
select f_nm from employee order by age;
select f_nm from employee order by age desc;
select *  from employee order by age;
select * from employee order by age, salary; -- second level sort will do incase of clash

-- Limit - to put a limit on the number of records to be fetched (filter - eliminate what is not required)
use practise;
select * from employee;
select * from employee limit 3;
select * from employee order by salary limit 2;
select * from employee order by salary desc limit 2;
select * from employee order by age limit 5;
select * from employee order by age,salary limit 4;
select id,f_nm,l_nm from employee order by id limit 1 ;
select * from employee order by id limit 2 offset 2;
----------------------------------------------
-- Aggregate functon
select count(*) from employee;
select count(location) from employee;
select count(distinct location) from employee;
select count(distinct location) as no_of_location from employee;
select count(f_nm) from employee where age >30;
select count(f_nm) from employee where age > 20 and age < 30;

select sum(salary) from employee;

select min(age) from employee;
select f_nm,l_nm from employee order by age limit 1;

--      Groupby and Having
-- where must be used before group by
select count(location) from employee;
select count(distinct location) from employee;
select location,count(*) from employee group by location;
select location,dept,count(*) from employee group by location ,dept;
select location,dept,age,count(*) from employee where age > 20 group by location , dept,age ;
-- having
select location,dept, count(*) as total from employee group by location,dept;

select location , count(*) as total from employee group by location having count(*) > 1; 
select dept ,count(*) from employee group by dept;
select dept, count(*) from employee group by dept having dept = 'IT';
select dept,count(*)  from employee where dept = 'it' group by dept;
---------------------------------------------------------------------------------------------
-- 4 constraints-------------------
-- DDL - create,alter,drop,truncate
-- DML - insert, update, delete
-- DQL - select

create table student(first_nm varchar(30) not null ,
last_nm varchar (30) not null,
age int not null,
cource_enrolled varchar(20) not null default 'Data science',
cource_fee int not null);

desc student;
select * from student;
insert into student (first_nm,last_nm,age,cource_fee) values
('daya','vallepu',25,95000),('madhavi','kumarri', 24,95000);

drop table student;
alter table student add id int not null;

use practise;
select * from student;

insert into student (id, first_nm, last_nm, age, cource_fee) values 
(1,'daya','vallepu',25,95000),
(2,'jagadeesh','k',24,95000),
(3,'Ashok','a',26,90000);
drop table student;

create table student(id serial primary key,first_nm varchar (30),
last_nm varchar (20), age int not null, cource varchar(20) not null default 'data scince');

desc student;
insert into student( first_nm, last_nm, age) values ('daya','vallepu',25),('priya','kumari',24);
select * from student;

drop table student;

use practise;
create table student (id int not null,first_nm varchar(20) not null,
last_nm varchar (20) not null,age int not null,cource_enrolled varchar (20) not null default 'data science',
primary key (id));
desc student;

insert into student (id, first_nm,last_nm,age) values(1, 'daya','vallepu',25),(2,'hemanth','vallepu',26);

select * from student;

-- composite primary key
create table sales_rep (rep_fname varchar (20) not null,rep_lastnm varchar(20) not null,
salary int not null);
select * from sales_rep;
insert into sales_rep values('latha','kumari',40000),('sindhu','priya',40000);
select * from sales_rep;
drop table sales_rep;

create table sales_rep(rep_fname varchar (20) not null,rep_lname varchar(20) not null,
salary int not null ,primary key(rep_fname,rep_lname));
describe sales_rep;
insert into sales_rep(rep_fname, rep_lname, salary) values('Anil', 'Sharma', 25000), ('Ankit', 'Verma', 30000), ('Sunil', 'Sharma', 25000);
select * from sales_rep;
insert into sales_rep(rep_fname, rep_lname, salary) values('Anil', 'Sharma', 25000), ('Ankit', 'Verma', 30000), ('Anil', 'Sharma', 25000); -- throw error

-- Auto increment
insert into student(id,first_nm,last_nm,age) values (2,'sandhya','kumari',24); -- duplicate entery , primary key doesnt allow duplicates
drop table student;
create table student (id int auto_increment primary key, f_nm varchar(20) not null,l_nm varchar (20) not null,age int not null);

insert into student(f_nm,l_nm,age) values('daya','vallepu',25);
select * from student;
insert into student(f_nm,l_nm,age) values('megha','vallepu',22);
alter table student auto_increment = 1001;
insert into student (f_nm,l_nm,age) values ('kaswi','k',05);


# [primary key could be on multiple columns called as composite key.]

-- Unique key 
use practise;
create table email_registration(f_nm varchar(30) not null,
l_nm varchar(20) not null,
email varchar(50) not null);
alter table email_registration add primary key(email);
desc email_registration;
alter table email_registration drop primary key ;
desc email_registration;
alter table email_registration add unique key(email);
alter table email_registration add primary key(f_nm,l_nm);
drop table email_registration;
create table email_registration(
f_name varchar(20),
l_name varchar(20),
email varchar(50) unique key,
primary key(f_name,l_name)
);
desc email_registration;
drop table email_registration;
-- DDL,DML,DQL

use practise;
select * from student;
select f_nm, l_nm from student where age= 25;
select * from student where f_nm like'____';
select * from student where binary f_nm = 'daya';
select * from student where binary f_nm = 'Daya'; -- it takes case sensitive

-- update statements 
set sql_safe_updates = 0;
update student set age = 24 where f_nm = 'daya';
select * from student;
delete from student; -- delete all rows

-- DDl (drop,alter,truncate)
alter table student drop column age;
desc student;
alter table student modify column f_nm varchar(40);
alter table student add column location varchar(30) default 'gudur';
select * from student;

desc employee;
alter table employee drop primary key;
alter table employee add primary key (f_nm,id);


use practise;
select * from employee;

create index idx_salary on employee(salary);

select * from employee where salary = 40000; -- When queries involve filtering or sorting by salary (e.g., SELECT * FROM employee WHERE salary > 50000;), the database can use this index to locate the relevant rows faster instead of scanning the entire employee table.
desc employee;

create table product (p_name varchar(20),quantity bigint);
desc product;
insert into product values('chairs',20000),('Tables',980000),('BOok cases',650000),('storage',5);
select * from product;
select p_name,quantity,case when quantity >10 then 'more than 10' when quantity < 10 then 'less than 10'
else 'equals to 10' end as quantitytext from product;
set sql_safe_updates = 0;
alter table product add quantity_text varchar (20) ;

update product set quantity_text = case
when quantity > 10 then 'more than 10'
when quantity < 10 then 'lwss than 10'
else 'equal to 10' end;
use practise; 
select * from product;
select * from product where quantity = 20000;
select * from product where quantity > 20000;
select * from product where quantity <= 20000;
select * from product where quantity <> 20000;
select * from product where quantity > 20000 or p_name ='chairs';
select * from product where quantity < 20000 and p_name = 'storage';
select * from product where quantity in (20000,5);
select * from product where quantity <= all(select quantity from product where quantity >20000);

use practise;
create table department (id int primary key, name varchar(20));
insert into department values(1,'IT'),(2,'HR');
select * from department;
create table emp1 (id serial primary key,name varchar(20),salary real,departmentID int not null,
foreign key(departmentID) references department (id));

drop table department;-- through error bcz emp1 table is the foreign key of department table
insert into emp1 (name, salary, departmentId)
values ('Ravi', 70000, 1), ('Ram', 90000, 1), ('Priya', 80000, 2), ('Mohan', 75000, 2), ('Shilpa', 90000, 1);
select * from emp1;
select * from department;
insert into emp1(name, salary, departmentid) values('Manoj',75000,3); -- Error Code: 1452. Cannot add or update a child row: a foreign key constraint fails (`practise`.`emp1`, CONSTRAINT `emp1_ibfk_1` FOREIGN KEY (`departmentID`) REFERENCES `department` (`id`))
-- Find out the names of Employees whose salary is less than the overall average salary?
select avg(salary) as avg_salary from emp1;
select * from emp1 where salary < 80000;
select * from emp1 where salary < all(select avg(salary) from emp1);


--                                         6a


use practise;
select * from employee;
create index idx_salary on employee(salary); -- index will use to retreive the data faster , its like an index of a book page to open exact page as faster
select * from employee where salary = 40000;

drop table product;
create table product (product_name varchar(20), quantity int);
insert into product(product_name, quantity) values ('Chairs',20),('Tables',5),('Bookcases', 10), ('Storage', 25);
select product_name, quantity, case when quantity > 10 then 'more than 10 '
when quantity < 10 then 'less than 10'
else 'equal to 10'
end as quantitytext from product;
-- saving the above data to a new table
select * from product;
create table product_1(product_name varchar (20),quantity int ,quantitytext varchar (20));
insert into product_1(product_name,quantity,quantitytext)
select product_name, quantity, case
when quantity > 10 then 'more than 10'
when quantity < 10 then 'less than 10'
else 'equal to 10'
end as quantitytext from product;

select * from product_1;

--                           6b   
use practise;
drop table emp1;
drop table department;
create table department( id int primary key, name varchar(20));
insert into department values(1, binary 'IT'),(2, binary 'HR');
select * from department;

create table emp1 (id serial primary key, name varchar(20), salary real,
departmentID int , foreign key (departmentID) references department(id));
desc emp1;
insert into emp1(name, Salary, departmentID) values
(binary'Ravi', 50000, 1),
(binary 'Daya',90000, 2),
(binary 'ram', 90000, 1),
(binary "Siva", 95000, 2),
(binary 'Poorni', 45000, 1);
select * from emp1;
 # Q. Find out the names of Employees whose salary is less than the overall average salary?
select name , salary from emp1 where salary < all(select avg(salary) from emp1);
use practise;
select departmentID,max(salary) from emp1 group by departmentid;
 -- Q. Show the name of the department.
select departmentid,department.name, max(salary) from emp1
inner join department on emp1.departmentID = department.id
group by departmentid, department.name;

# Q. show the name of the employee also.
select departmentid, department.name, emp1.name, max(salary) from emp1 inner join department
on emp1.departmentid = department.id group by departmentid,department.name,emp1.name;
-- This doesn't work as we are now creating groups on the combination of Department and Employee.

select department.name, emp1.name, salary from emp1 
inner join department on emp1.departmentid = department.id
where (departmentid, salary) in (select departmentid, max(salary) as salary from emp1 group by departmentid);

-- Q. Selecting the second-highest salary of an employee
select salary as second_salary from emp1 order by salary desc  limit 1 offset 1;
select max(salary) from emp1 where salary < (select max(salary) from emp1);

--                            6c.   Joins
use practise;
create table courses (name varchar(20), cource varchar(20)); 
insert into courses values ('BBB', 'Tablue'),('CCC', 'Python'),('DDD', 'Data Analytics'),('DDD', 'SQL');
select * from courses;

create table students (name varchar(20), age int );
insert into students values('AAA' , 22),("BBB", 24),('CCC', 25),('DDD', 30);
select * from students;

-- Inner joins:
select name, cource, age from students inner join courses on name = name; 
select students.name, age,cource from students inner join courses on students.name = courses.name;
-- left join
select students.name, age, cource from students left join courses on students.name = courses.name;

-- right join
select * from students right join courses on students.name = courses.name;

-- Full join
select * from students full join courses on students.name = courses.name;
use practise;
-- constraints
create table school(name varchar(25), schoolname varchar(30) default '360 DigiTMG',
age int ,check(age >= 10));
desc school;
insert into school(name, age) values (binary 'daya',25),
(binary 'Ravi', 26),(binary 'Ram', 42);
select * from school;
insert into school(name, age) values ('priya', 8); -- new row for relation "school" violates check constraint "school_age_check"

-- TimeStamp and Date data types
create table emp(id serial primary key,
name varchar (20) not null,
dept varchar (10) not null,
date_of_joining timestamp not null default current_timestamp,
status varchar(10) default 'active',
salary real not null,
last_updated timestamp not null default now());
desc emp;
select * from emp;
insert into emp (name, dept, salary) values ('Ravi Kiran', 'HR', 40000.00), 
('Priya Darshini', 'IT', 25000.00),('Mohan Bhargav', 'Finance', 30000.00);

-- 6d
create table employees(
employee_id serial,
employee_name varchar (100),
department varchar(20),
salary decimal(10,2));
select * from employees;
INSERT INTO employees (employee_name, department, salary)
VALUES
    ('John Doe', 'IT', 50000.00),
    ('Jane Smith', 'HR', 60000.00),
    ('Bob Johnson', 'Finance', 55000.00),
    ('Alice Brown', 'IT', 45000.00),
    ('Mark Davis', 'Finance', 65000.00),
    ('Sarah Adams', 'HR', 55000.00),
    ('Mike Wilson', 'IT', 48000.00),
    ('Emily Clark', 'Finance', 60000.00),
    ('David Anderson', 'IT', 52000.00),
    ( 'Jessica Lee', 'HR', 58000.00),
    ( 'Daniel Harris', 'Finance', 56000.00),
    ( 'Sophia Taylor', 'IT', 49000.00);
    
select * from employees;

-- creating the first view
create view vw_it_employees as select employee_id, employee_name, salary from employees where department = 'IT';
drop view vw_it_employees;
select * from vw_it_employees;
-- creating second view 
create view vw_hr_employees as select employee_id, employee_name, salary from employees where department = 'HR';
select * from vw_hr_employees;
-- creating the third view
create view vw_finance_employees as select employee_id, employee_name, salary from employees where department = 'Finance';
select * from vw_finance_employees;

-- Create a view for employees with names starting with 'J'
CREATE VIEW vw_j_employees AS
SELECT employee_id, employee_name, department, salary
FROM employees
WHERE employee_name LIKE 'J%';

select * from vw_j_employees;

-- Create a view for employees with names containing 'o'
CREATE VIEW vw_o_employees AS
SELECT employee_id, employee_name, department, salary
FROM employees
WHERE employee_name LIKE '%o%';
select * from vw_o_employees;
use practise;

-- unions in module 7 
create table aug2016 (day int, customer varchar(20),
purchases int, type varchar(20));

create table sep2016 (day int, customer varchar(20),
purchases real, type varchar(20));

create table oct2016 (day int, customer varchar(20),
purchases real, type varchar(20));

-- import the data
select * from oct2016 union select * from aug2016 union select * from sep2016;

insert into aug2016(day, customer, purchases, type)
values (1, 'john Doe', 500, 'Mobile'),
(2, 'jane smith', 600, 'furniture');
INSERT INTO sep2016(Day, customer, purchases, type)
VALUES
    (1, 'John', 10.20, 'A'),
    (2, 'Smith', 600.30, 'C');

 INSERT INTO oct2016(Day, Customer, Purchases, Type)
VALUES
    (1, 'Sharat', 100, 'ABC'),
    (2, 'Smith', 30, 'D');

select * from Aug2016 UNION select * from Sep2016 UNION select * from Oct2016;
select * from students;

-- Triggers
show databases;
drop database 360db;
create database 360db;
use 360db;
create table student_info (
stud_id int not null,
stud_code varchar(15) default null,
stud_name varchar(25) default null,
subject varchar(35) default null,
marks int default null,
phone varchar(20) default null,
primary key(stud_id));
insert into student_info values(101, 01, 'shiv', 'Maths', 50, '966644582');
insert into student_info values(102, 02, 'shivi', 'Maths', 50, '966677582');
select * from student_info;

create table student_detail (stud_id int not null primary key,
stud_code varchar(20) default null,
stud_name varchar(20) default null,
subject varchar(20) DEfault null,
marks int default null,
phone varchar(15) default null,
lasinserted Time
);
select * from student_detail;

delimiter $$
create trigger after_insert_details 
after insert on student_info for each row
begin
insert into student_detail
values(new.stud_id, new.stud_code, new.stud_name,new.subject,new.marks,new.phone, curtime());
end $$
delimiter ;
drop trigger after_insert_details;
select * from student_info;
select * from student_detail;
insert into student_info(stud_id,stud_code,stud_name,subject,marks,phone) values(103, 031, 'Amar', 'Datascience', 80, '66644582');
insert into student_info values(104, 032, 'Amer', 'Datascience', 90, '66644583');
set sql_safe_updates = 0;
-- [alter table  student_detail add phone varchar(20) default null;]
select * from student_info;
select * from student_detail;

delimiter // 
create trigger before_inserted_details
before insert on student_info 
for each row
begin
insert into student_detail
values(new.stud_id,new.stud_code, new.stud_name, new.subject, new.marks, new.phone, curtime());
end // 
delimiter ;

insert into student_info values(105, 033, 'Aditya', 'Datascience', 90, '66644584');
# This will throw error as the After trigger is active and trying to update the table.

drop trigger after_insert_details;
select * from student_info;
select * from student_detail;


-- 8a 
use practise;
show databases;
create database dataanalytics_db;
use dataanalytics_db;
-- loading a external file
create table education (datasrno int, workex int, gmat int);
desc education;
insert education values(1, 10, 700),(2, 20, 800);
select * from education;
load data infile "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/education.csv"
into table education
fields terminated by ','
enclosed by '"'
lines terminated by '\n'
ignore 1 rows;

select * from education;

--                                             mean 
select avg(workex) as mean_workes from education;

--                                             median
use practise;

select workex as median_experience from (
select workex, row_number() over (order by workex) as row_num,
count(*) over () as total_count from education)
as subquery
where row_num = (total_count + 1) / 2 or row_num = (total_count + 2)/2;

--                                            mode
select workex as mode_exp from (
select workex, count(*) as frequency from education
group by workex
order by frequency desc
limit 1 
) as subquery;

--                                         variance
select variance(workex) as workex_var from education;

--                                     std deviation
select stddev(workex)  as workex_stddev from education;
--                                  Range

select max(workex) - min(workex) as exp_range from education;

--                               skewness 
select (sum(power(workex - (select avg(workex) from education),3))/
(count(*) * power((select stddev(workex) from education),3))) as skewness from education;

--                              kurtosis

select (sum(power(workex - (select avg(workex) from education),4))/
(count(*) * power((select stddev(workex) from education),4))-3) as kurtosis from education;

--                                              winsorization

WITH ranked_data AS (
    SELECT 
        workex,
        NTILE(100) OVER (ORDER BY workex) AS percentile_rank
    FROM education
),
bounds AS (
    SELECT 
        MIN(CASE WHEN percentile_rank = 5 THEN workex END) AS lower_bound,
        MAX(CASE WHEN percentile_rank = 95 THEN workex END) AS upper_bound
    FROM ranked_data
)
SELECT 
    CASE 
        WHEN workex < (SELECT lower_bound FROM bounds) THEN (SELECT lower_bound FROM bounds)
        WHEN workex > (SELECT upper_bound FROM bounds) THEN (SELECT upper_bound FROM bounds)
        ELSE workex
    END AS winsorized_workex
FROM education;
--                             set complted 
--                           another set started again for revision

--       modele- 1
show databases;
use practise;
show tables; -- to see the list of tables in a current  database you used
create database practise2;
use  practise2;
create table employee(first_name varchar(20) not null, -- crating a table
middle_name varchar(20),
last_name varchar(20) not null,
age int not null,
salary int not null,
location varchar(20) not null default'Hyderabad');
desc employee; -- describing a table
insert into employee values('daya', 'vallepu', 'daya', 25, 95000, 'Banglore');
select * from employee;

create table education(datasrno int, workex int, gmat int);
 insert into education values(1, 10, 700), (2, 20, 740);
 select * from education;

load data infile  "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/education.csv" -- getting external file
into table education
fields terminated by ','
enclosed by '"'
lines terminated by "\n"
ignore 1 rows;
select * from education;

set sql_safe_updates = 0;
delete from education where datasrno in (1, 2) ; -- perticular rows deleted

drop table education;
use practise2;
create table employee(id serial primary key,
f_nm varchar(20) not null,
l_nm varchar(20) not null,
age int not null,
locaton varchar(20) not null default 'hyderabad',
dept varchar(20) not null);

alter table employee add salary real not null;
desc employee;

insert into employee (f_nm, l_nm, age, dept, salary) values
('ravi','kiran', 25, 'hr', 30000.00),
('priya', 'darshini', 28, 'hr', 32000.00),
('mohan', 'bhargavi', 35, 'IT', 40000.00);
insert into employee (f_nm, l_nm, age, locaton, dept, salary) values
('Akhil', 'K', 26, 'Bangalore', 'IT', 42000.00),
('Raja', 'Roy', 35, 'Bangalore', 'IT', 60000.00),
('Shilpa', 'Sharma', 40, 'Chennai', 'IT', 44000.00);

select * from employee;

--  Distinct - unique vvvalues no repetetion
use practise2;
select locaton from employee;
select distinct locaton from employee;
select distinct dept from employee;
select count(distinct dept) from employee;

-- order by
select f_nm from employee;
select f_nm from employee order by F_nm;
select f_nm from employee order by f_nm desc;
select f_nm from employee order by age;
select f_nm from employee order by age desc;
select * from employee order by age, salary; -- second level sort will happen incase of clash

-- Limit- to put a limit on the number of records to be fetched 
select * from employee limit 3;
select * from employee order by salary limit 3 ;
select * from employee order by salary desc limit 3;
select * from employee order by age limit 5;
select * from employee order by age , salary limit 4 ;
select id, f_nm l_nm from employee order by id limit 1 offset 0;
select id, f_nm, l_nm from employee order by id limit 3 offset 3;-- when we apply offset , it will take that point to consideration

-- Aggregate function
use practise2;
select count(*)from employee;
select count(locaton) from employee;
select count(distinct locaton) from employee;
select count(distinct locaton) as no_of_location from employee;
select count(f_nm) from employee where age > 20;
select count(f_nm) from employee where age > 25 and age < 35;
select sum(salary) from employee;
select avg(salary) from employee;
select min(age) from employee;
select * from employee where age limit 1 ;
-- Groupby and Having

select count(locaton) from employee;
select locaton,count(*) from employee group by locaton;
select locaton, dept, count(*) from employee group by locaton,dept;
select locaton, dept, count(*) from employee where age < 30 group by locaton, dept;
select locaton, count(*) as total from employee group by locaton having count(*) > 1 ;
select locaton, count(*) from employee group by locaton;
select locaton, count(*) from employee group by locaton having locaton = 'hyderabad';

use practise2;
create table student(first_name varchar(20) not null, last_name varchar(20) not null,
age int not null,
coutce_enrolled varchar(20) not null default' data analytics',
cource_fee int not null);
desc student;
insert into student(first_name, last_name, age, cource_fee) values 
('Madhavi', 'Kumari', 24, 40000);

select * from student;
drop table student;
set sql_safe_updates = 0;
alter table student add column id int ;
select * from student;
drop table student;
create table student(
id int,
first_name varchar(20) not null,
last_name varchar(20) not null,
age int not null,
course_enrolled varchar(20) not null default 'Data Analytics',
course_fee int not null
);

desc student;
insert into student(id, first_name, last_name, age, course_fee) values (1, 'Madhavi', 'Kumari', 24, 40000);

insert into student(id, first_name, last_name, age, course_fee) values (1, 'Madhavi', 'Kumari', 24, 40000);
select * from student;

insert into student(id, first_name, last_name, age, course_fee) values (null, 'Madhavi', 'Kumari', 24, 40000);

drop table student;
use practise2;
create table student ( id int primary key, first_name varchar(20) not null, last_name varchar(20) not null,
age int not null, 
cource_enrolled varchar(20) not null default 'Data Analytics',
cource_fee int not null);
select * from student ;
desc student;

insert into student(id, first_name, last_name, age, cource_fee) values (1, 'Madhavi', 'Kumari', 24, 40000);
insert into student(id, first_name, last_name, age, cource_fee) values (2, 'Madhavi', 'Kumari', 24, 40000);
insert into student(id, first_name, last_name, age, cource_fee) values (2, 'Madhavi', 'Kumari', 24, 40000);
select * from student;

-- Composite Primary key
create table sales_rep (rep_fname varchar(20) not null,
rep_lname varchar(20) not null,
salary int not null); 

insert into sales_rep values('Anil', 'Sharma', 25000),
('Ankit', 'varma', 30000),
('Anil', 'sharma', 25000);
select * from sales_rep;
drop table sales_rep;
create table  sales_rep ( rep_fname varchar(20) not null,
rep_lname varchar(20) not null,
salary int not null,
primary key(rep_fname, rep_lname));

desc sales_rep;

insert into sales_rep(rep_fname, rep_lname, salary) values('Anil', 'Sharma', 25000), ('Ankit', 'Verma', 30000), ('Anil', 'Sharma', 25000);
--- will throw an error

insert into sales_rep(rep_fname, rep_lname, salary) values('Anil', 'Sharma', 25000), ('Ankit', 'Verma', 30000), ('Sunil', 'Sharma', 25000);
select * from sales_rep;
-- Autoincrement
INSERT INTO STUDENT(ID, FIRST_NAME, LAST_NAME, AGE,COURCE_ENROLLED, COURCE_FEE) VALUES(2, 'SANDHYA','DEVI',28,'DATA SCIENCE',50000);
-- ABOVE QUERY GIVES ERROR BECAUSE DUPLICATE ENTRY OF PRIMARY KEY

DROP TABLE STUDENT;
CREATE TABLE STUDENT(ID INT NOT NULL AUTO_INCREMENT,
FIRST_NAME VARCHAR(20) NOT NULL,
Last_name varchar(20) not null,
age int not null,
course_enrolled varchar(20) not null default (binary 'Data Analytics'),
cource_fee int not null, primary key (id));

select * from student;
desc student;
insert into student(first_name, last_name, age, course_enrolled, cource_fee) values ('Sandhya', 'Devi', 28, 'Data Science', 50000), ('Priya', 'Darshini', 25, 'Data Science', 50000);
insert into student(first_name, last_name, age, cource_fee) values ('Ravi', 'Mittal', 28, 30000), ('Akhil', 'K', 25, 30000);
select * from student;

create table identification(id int auto_increment, name varchar(20), primary key (id));
alter table identification auto_increment = 1001;
insert into identification(name) values(binary 'Daya'),('Ravi'),('vamsi');
select * from identification;


--                                                   Daily practise A & D 
# 22/02/2025

# How do you find the third highest salary from an employee table without using LIMIT?
use practise;
select max(salary) as highest_salary from employee where salary  
< (select max(salary) from employee where salary < 
(select max(salary) from employee)); 

# Write a query to display the employee name and their manager's name in the same row.
use practise2;
select * from employee;
select f_nm, manager from employee;
# or 
select e.f_nm as employee, m.manager as manager from employee e
left join employee m on e.id = m.id;

# How do you get the total sales for each month from a sales table?

use supermart_db;
SELECT DATE_FORMAT(STR_TO_DATE(order_date, '%d-%m-%Y'), '%m-%Y') AS month_year,  
       SUM(sales) AS total_sales  
FROM sales  
GROUP BY month_year  
ORDER BY STR_TO_DATE(month_year, '%m-%Y');


# 24/02/2025

-- Write an SQL query to retrieve all employees’ names and their department names using INNER JOIN.
-- Tables: Employees(id, name, department_id) and Departments(id, name).
use 360DigiTMG;
select a.employee_name as employee_name, d.name as department_name 
from employees a inner join department d on a.employee_id = d.id;

# GROUP BY and HAVING:
# Question: Write an SQL query to find departments with more than 5 employees.
# Table: Employees(id, name, department_id).

select department from employees group by department having 3 < count(*);

-- Question: Write a query to find employees whose salary is higher than the average salary of their department.
-- Table: Employees(id, name, department_id, salary).
select department,employee_name, salary from employees e where salary > 
(select avg(salary) from employees where department = e.department);

-- Question: Write an SQL query to assign a rank to each employee based on their salary within each department using the RANK() function.
SELECT 
    department, 
    employee_name, 
    salary, 
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank
FROM Employees;

select department, employee_name, salary,
rank() over (partition by department order by salary) as salary_rank from employees;
 
-- Self-Join:
-- Question: Write an SQL query to find all employees who have the same manager.
-- Table: Employees(id, name, manager
use practise2;
SELECT e1.f_nm AS employee1, e2.f_nm AS employee2, e1.manager
FROM Employee e1
JOIN Employee e2 ON e1.manager = e2.manager
WHERE e1.id != e2.id
ORDER BY e1.manager, e1.f_nm, e2.f_nm;

use practise;
select max(salary) as highest_salary from employee  where salary < (select max(salary) from employee);
select distinct salary from employee order by salary desc limit 1 offset 2;
select f_nm, count(*) from employee group by f_nm having count(*) > 1; -- Finding duplicates
select dept, sum(salary) as total_salary from employee  -- top 3 Highest salary
group by dept order by total_salary desc limit 3;

use 360digitmg;
select * from employees;
-- give the name and salry of second person highest salary
select employee_name, salary as mx_salary from employees
where salary < (select  max(salary) from employees) order by salary desc limit 1;

create database ele_price_forecast;
use ele_price_forecast;
select * from date_time_mcp_imputed where Datetime = '2024-07-15 01:00:00';

(SELECT * FROM date_time_mcp_imputed  
 WHERE Datetime <= '2024-12-05 20:00:00'  
 ORDER BY Datetime DESC  
 LIMIT 5)  

UNION ALL  

(SELECT * FROM date_time_mcp_imputed  
 WHERE Datetime > '2024-12-05 20:00:00'  
 ORDER BY Datetime ASC  
 LIMIT 4);

drop database ele_price_forecast;

use practise;
select max(salary) from employee where salary not in (select max(salary) from employee);

select max(salary) from employee where salary < (select max(salary) from employees);
select * from employees where salary
