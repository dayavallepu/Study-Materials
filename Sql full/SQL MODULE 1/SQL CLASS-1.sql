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


















