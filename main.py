# print(f"3 + 2 = {3 + 2}")
# print(f"3 - 2 = {3 - 2}")
# print(f"3 * 2 = {3 * 2}")
# print(f"3 / 2 = {3 / 2}")
# print(f"3^2 = {3 ** 2}")

# com = 1 + 2j
# print(com)
# com1 = complex(1,2)
# print(com1)
# print(complex(4,-1))

# print('he' + 'llo')
# print('he'*5)
# print('hello\nworld')
# print('hello\tworld')

# str = "abcde"
# print(str[2])
# print(str[0:5:2])

#### FOR LOOPS

# cau 1
# name = "Van Pham Gia Huy"
# for x in name:
#     print(x, end=' ')

# cau 2
# for i in range(1, 11, 2):
#     print(i)

# cau 3a
# sum = 0
# for i in range(1, 11, 2):
#     sum += i
# print(sum)
# c2
# print(sum(range(1, 11, 2)))

# cau 3b
# sum = 0
# for i in range(1, 7):
#     sum += i
# print(sum)
# c2
# print(sum(range(1, 7)))

# cau 4
# mydict = {
#     "a" : 1,
#     "b" : 2,
#     "c" : 3,
#     "d" : 4
# }

# 4a
# for i in mydict:
#     print(i)
# 4b
# for i in mydict.values():
#     print(i)
# 4c
# for i in mydict.items():
#     print(i)

# cau 5
# courses = [131,141,142,212]
# names = ["Maths", "Physics", "Chem", "Bio"]

# mylist = []
# for i in range(len(courses)):
#     tuple = (courses[i], names[i])
#     mylist.append(tuple)
# print(mylist)
# c2
# print(list(zip(courses, names)))

# cau 6
# str = "jabbawocky"
# check = ['u', 'e', 'o', 'a', 'i']

# 6a
# count = 0
# for i in str:
#     if i not in check:
#         count += 1
# print(count)
# c2
# print(len([i for i in str if i not in check]))

# 6b
# count = 0
# for i in str:
#     if i in check:
#         continue
#     count += 1
# print(count)

# cau 7
# for a in range(-2,3):
#     try:
#         print(f"10 / {a} = {10/a}")
#     except ZeroDivisionError:
#         print("can't divided by zero")

# cau 8
# ages = [23,10,80]
# names = ["Hoa","Lam","Nam"]

# mylist = list(zip(ages, names))
# sort = sorted(mylist, key = lambda x : x[0])
# print(sort)

# cau 9
# 9a
# file = open("firstname.txt", mode='r')

# 9b
# for line in file:
#     print(line, end='')

# 9c
# content = file.read()
# print(content)

# file.close()

#### FUNCTION

# cau 1
# def my_sum(a, b):
#     return a + b

# print(my_sum(3,4))

# cau 2
import numpy as np

def check_rank(lst):
    return np.linalg.matrix_rank(lst)

def check_shape(lst):
    return np.shape(lst)

# matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
# vector = [1,2,3]

# print(check_rank(matrix))
# print(check_shape(vector))

# cau 3
# new_matrix = matrix + 3
# print(new_matrix)

# cau 5
# x = np.array([2,7])
# norm_x = np.linalg.norm(x)
# normalized_x = x / norm_x

# cau 8
# A = np.array([[2,4,9], [3,6,7]])

# 8a
# print(check_rank(A))
# print(check_shape(A))

# 8b
# print(A[1][2])

# 8c
# print(A[:,1])

# cau 9
# matrix = np.random.randint(-10, 11,(3,3))
# print(matrix)

# cau 10
# matrix = np.eye(3)
# print(matrix)

# cau 11
# matrix = np.random.randint(1,10,(3,3))
# print(matrix)

# 11a
# print(np.trace(matrix))

# 11b
# sum = 0
# count = 0
# for i in range(len(matrix)):
#     sum += matrix[count][count]
#     count += 1
# print(sum)

# cau 12
# matrix = np.diag([1,2,3])
# print(matrix)

# cau 13
# A = np.array([[1,1,2],[2,4,-3],[3,6,-5]])
# print(round(np.linalg.det(A)))

# cau 14
# a1=[1,-2,-5]
# a2=[2,5,6]

# matrix = np.column_stack((a1,a2))
# print(matrix)

# cau 15
import matplotlib.pyplot as plt

# y = np.arange(-5, 6, 1)
# y_squared = y**2

# plt.figure()
# plt.plot(y, y_squared, marker='o', linestyle='-')
# plt.xlabel('y')
# plt.ylabel('y^2')
# plt.title('Plot of y^2')
# plt.grid(True)
# plt.show()

# cau 16
# values = np.linspace(0, 32, 4)
# print(values)

# cau 17
# x = np.linspace(-5, 5, 50)
# y = x**2

# plt.figure()
# plt.plot(x, y, label='$y = x^2$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of y = x^2')
# plt.grid(True)
# plt.show()

# cau 18
# x = np.linspace(-2, 2, 100)
# y = np.exp(x)

# plt.figure()
# plt.plot(x, y, label='$y = e^x$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of y = exp(x)')
# plt.grid(True)
# plt.show()

# cau 19
# x = np.linspace(0.1, 5, 100)
# y = np.log(x)

# plt.figure()
# plt.plot(x, y, label='$y = log(x)$')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Plot of y = log(x)')
# plt.grid(True)
# plt.show()

# cau 20
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
x1 = np.linspace(-2, 2, 100)
y1 = np.exp(x1)
y2 = np.exp(2 * x1)

axs[0].plot(x1, y1, label='$y = e^x$', linestyle='-')
axs[0].plot(x1, y2, label='$y = e^{2x}$', linestyle='--')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Exponential Functions')
axs[0].legend()
axs[0].grid(True)

x2 = np.linspace(0.1, 5, 100)  
y3 = np.log(x2)
y4 = np.log(2 * x2)

axs[1].plot(x2, y3, label='$y = log(x)$', linestyle='-')
axs[1].plot(x2, y4, label='$y = log(2x)$', linestyle='--')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Logarithmic Functions')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()