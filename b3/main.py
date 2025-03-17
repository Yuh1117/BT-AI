# cau 1
import mglearn.datasets
from sklearn.datasets import load_iris

import mglearn.data

iris = load_iris()
iris_data = iris.data
# print(iris_data[:5])

# cau 2
# print(iris.target[:5])
# print(iris.target_names)

# cau 3
# X chieu dai dai
# Y chieu rong dai
# moi loai khac mau

from matplotlib import pyplot as plt

# sepal_length = iris_data[:,0]
# sepal_width = iris_data[:,1]

# x1 = sepal_length[iris.target == 0]
# y1 = sepal_width[iris.target == 0]

# x2 = sepal_length[iris.target == 1]
# y2 = sepal_width[iris.target == 1]

# x3 = sepal_length[iris.target == 2]
# y3 = sepal_width[iris.target == 2]

# plt.scatter(x1, y1, label='Setosa', c='red')
# plt.scatter(x2, y2, label='Versicolor', c='blue')
# plt.scatter(x3, y3, label='Virginica', c='green')
# plt.xlabel = 'chieu dai dai'
# plt.ylabel = 'chieu rong dai'
# plt.title = 'title'
# plt.legend()
# plt.show()

# cau 4
# from sklearn.decomposition import PCA

# pca = PCA(n_components=3)
# iris_pca = pca.fit_transform(iris.data)
# print(iris_pca[:5])

# cau 5
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=140, test_size=10, random_state=42)
# print(x_train.shape)

# cau 6
from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(x_train, y_train)
# y_pred = knn.predict(x_test)
# print(y_pred)

# cau 7
# from sklearn.metrics import accuracy_score

# accuracy = accuracy_score(y_test, y_pred)
# print(y_test)
# print(y_pred)
# print(accuracy)

# cau 8
import numpy as np

# X = iris.data[:, [0,1]]
# y = iris.target

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X, y)

# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.contour(xx, yy, Z, alpha=0.3)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
# plt.xlabel("chieu dai dai hoa")
# plt.ylabel("chieu rong dai hoa")
# plt.title("ranh gioi quyet dinh voi KNN")
# plt.show()

# cau 9
# from sklearn.datasets import load_diabetes

# diabetes = load_diabetes()
# print(diabetes.data[:5])

# cau 10
# x_train = diabetes.data[:422]
# x_test = diabetes.data[422:]
# y_train = diabetes.target[:422]
# y_test = diabetes.target[422:]

# print(x_train.shape)

# cau 11
from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(x_train, y_train)

# cau 12
# coefficients = lr.coef_
# print(coefficients)

# cau 13
# y_pred = lr.predict(x_test)
# print(y_pred)

# cau 14
# from sklearn.metrics import r2_score

# r2 = r2_score(y_test, y_pred)
# print(r2)

# cau 15
# x_train_age = x_train[:, [0]]
# x_test_age = x_test[:, [0]]
# lr_age = LinearRegression()
# lr_age.fit(x_train_age, y_train)
# y_pred_age = lr_age.predict(x_test_age)
# print(y_pred_age)

# cau 16
# for i in range(10):
#     x_train_simple = x_train[:, [i]]
#     x_test_simple = x_test[:, [i]]
#     lr_simple = LinearRegression()
#     lr_simple.fit(x_train_simple, y_train)
#     y_pred_simple = lr_simple.predict(x_test_simple)
    
#     plt.figure()
#     plt.scatter(x_test_simple, y_test, color='blue', label='Thuc te')
#     plt.plot(x_test_simple, y_pred_simple, color='red', label="Du doan")
#     plt.xlabel(f'Dac trung {i + 1}')
#     plt.ylabel(f'Hoi quy tuyen tinh dac trung {i + 1}')
#     plt.legend()
#     plt.show()

# cau 17
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
# print(breast_cancer.keys())

# cau 18
import pandas as pd

# print(breast_cancer.data.shape)
# target_series = pd.Series(breast_cancer.target)
# benign_count = target_series.value_counts()[1]
# malignat_count = target_series.value_counts()[0]
# print(benign_count)
# print(malignat_count)

# cau 19
# x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(breast_cancer.data, breast_cancer.target, 
#                                                             test_size=0.2, random_state=42)
# train_scores = []
# test_scores = []
# for k in range(1, 11):
#     knn_1 = KNeighborsClassifier(n_neighbors=k)
#     knn_1.fit(x_train_1, y_train_1)
#     train_scores.append(knn_1.score(x_train_1, y_train_1))
#     test_scores.append(knn_1.score(x_test_1, y_test_1))

# plt.plot(range(1, 11), train_scores, label='Do chinh xac cua tap huan luyen')
# plt.plot(range(1, 11), test_scores, label='Do chinh xac cua tap kiem thu')
# plt.title("Do chinh xac")
# plt.xlabel("So lang gieng")
# plt.ylabel("Do chinh xac")
# plt.legend()
# plt.show()

# cau 20
import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# X, y = mglearn.datasets.make_forge()

# logreg = LogisticRegression().fit(X, y)
# print("Do chinh xac Logistic Regression: ", logreg.score(X, y))

# svc = LinearSVC().fit(X ,y)
# print("Do chinh xac Linear SVC: ", svc.score(X, y))

# cau 21
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(resize=0.4)

# print(faces.DESCR)


# cau 22
# print(faces.images.shape)

# cau 23
# def plot_faces(images, n_row=2, n_col=5):
#     plt.figure(figsize=(2 * n_col, 2.5 * n_row))
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i], cmap='gray')
#         plt.axis('off')
#     plt.show()

# plot_faces(faces.images)

# cau 24
from sklearn.svm import SVC

svc = SVC(kernel='linear')

# cau 25
# X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, 
#                                                             test_size=0.25, random_state=42)
# print("Kich thuoc tap huan luyen", X_train.shape)
# print("Kich thuoc tap kiem tra", X_test.shape)

# cau 26
from sklearn.model_selection import cross_val_score

# def evaluate_cross_validation(model, X, y, k=5):
#     scores = cross_val_score(model, X, y, cv=k)
#     print(f'Do chinh xac K-fold (k={k}): {scores.mean(): .2f} (+/- {scores.std() * 2: .2f})')

# cau 27
# def train_and_evaluate(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     print("Do chinh xac tap huan luyen: ", train_score)
#     print("Do chinh xac tap kiem tra: ", test_score)

# cau 28
# evaluate_cross_validation(svc, faces.data, faces.target)

# train_and_evaluate(svc, X_train, X_test, y_train, y_test)

# cau 29
def create_glasses_target(target):
    np.random.seed(42)
    glasses_target = np.random.randint(0, 2, size=len(target))
    return glasses_target

faces_glasses_target = create_glasses_target(faces.target)
print("Mang muc tieu moi (10 gia tri dau tien): ", faces_glasses_target[:10])

# cau 30
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces_glasses_target, 
                                                            test_size=0.25, random_state=42)

svc_2 = SVC(kernel='linear')

svc_2.fit(X_train, y_train)

# cau 31
def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f'Do chinh xac K-fold (k={k}): {scores.mean(): .2f} (+/- {scores.std() * 2: .2f})')

evaluate_cross_validation(svc_2,X_train,y_train,5)

# cau 32
X_eval = faces.data[30:40]
y_eval = faces_glasses_target[30:40]

X_train_remaining = np.concatenate((faces.data[:30], faces.data[40:]))
y_train_remaining = np.concatenate((faces_glasses_target[:30], faces_glasses_target[40:]))

svc3 = SVC(kernel='linear')
svc3.fit(X_train_remaining, y_train_remaining)

acurracy = svc3.score(X_eval, y_eval)
print("Do chinh xac tren tap 10 anh", acurracy)

# cau 33
y_pred = svc3.predict(X_eval)

eval_faces = [np.reshape(a, (50, 37)) for a in X_eval]

def plot_faces(images, prediction, n_col=10):
    plt.figure(figsize=(2 * n_col, 2.5))
    for i in range(len(images)):
        plt.subplot(1, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Pred: {prediction[i]}')
        plt.axis('off')

    plt.show()

plot_faces(eval_faces, y_pred)

for i in range(len(y_eval)):
    if y_eval[i] != y_pred[i]:
        print(f'Anh o chi so {i + 30} bi pha loai sai. Thuc tue: {y_eval[i]}, du doan: {y_pred[i]}')