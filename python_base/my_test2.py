'''
@author: xuxiangfeng 
@date: 2020/1/10
'''
from collections.abc import Iterator, Iterable
a = [3,5,6,7,8]

# for i in a:
#     print(a)

# a_iter = iter(a)
#
# while True:
#     try:
#         i = next(a_iter)
#         print(i)
#     except StopIteration:
#         break


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say(self):
        print("person saying")


class Student(Person):
    def __init__(self, name, age):
        super(Student, self).__init__(self, name, age)

    # def say(self):



    def show(self):
        print("hello")



if __name__ == '__main__':
    s = Student()
    print(issubclass(Student, Person))




