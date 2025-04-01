from cProfile import label

import numpy as np
from matplotlib.lines import lineStyles
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split

# buoc 1
digits = datasets.load_digits()
X = digits.data
y = digits.target


# print(digits.data)
# print(digits.target)


# buoc 2 Dinh nghia ham de ve cac chu so
def plot_digits(data, labels, num_rows=2, nums_cols=5, title="Chữ số viết tay"):
    fig, axes = plt.subplots(num_rows, nums_cols, figsize=(10, 4))

    for i, ax in enumerate(axes.ravel()):
        if i < len(data):
            ax.imshow(data[i].reshape(8, 8), cmap='gray')
            ax.set_title(f"Nhãn {digits.target[i]}")
            ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


plot_digits(X, y, num_rows=2, nums_cols=5)


# buoc 3 Dinh nghia ham de ve bieu do phan tan voi PCA
def plot_pca_scatter(X, y, title="PCA của tập dữ liệu chữ số viết tay"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(digits.data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.6, edgecolors='w')
    plt.colorbar(scatter, label="lớp chữ số/cụm")
    plt.title("PCA của chữ số viết tay")
    plt.xlabel('Thành phần chính 1')
    plt.ylabel("Thành phần chính 2")
    plt.title(title)
    plt.grid(True)
    plt.show()

    return X_pca, pca


plot_pca_scatter(X, y)

# buoc 4 Thuc hien PCA va ve ket qua
print("\nBiểu đồ phân tán 2D sau khi áp dụng PCA: ")

X_pca, pca_model = plot_pca_scatter(X, y)


# buoc 5 Ve cac thanh phan PCA duoi dang hinh anh 8x8
def plot_pca_components():
    pca = PCA(n_components=10)
    pca.fit(X)
    components = pca.components_

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.ravel()):
        if i < len(components):
            ax.imshow(components[i].reshape(8, 8), cmap='gray')
            ax.set_title(f'Thành phần {i + 1}')
            ax.axis('off')
    plt.suptitle("10 thành phần chính dưới dạng hình ảnh 8x8")
    plt.tight_layout()
    plt.show()


print("\n Hiển thị 10 thành phần chính dưới dạng hình ảnh 8x8")
plot_pca_components()

## buoc 6 Thuc hien phan cum bang k-means

# buoc 7 Chi du lieu thanh tap huan luyen va kiem tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Kích thước tập huấn luyện", X_train.shape)
print("\n Kích thước tập kiểm tra", X_test.shape)


# buoc 8 Thu nghiem voi tham so n_init cua k-means
def experiment_with_n_init(X_train, y_train, n_clusters=10):
    n_init_values = [1, 5, 10, 20]
    ari_scores = []

    print("\n Thử nghiệm với các giá trị n_init khác nhau cho k-means: ")
    for n_init in n_init_values:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans.fit(X_train)
        cluster_labels = kmeans.labels_
        ari = adjusted_rand_score(y_train, cluster_labels)
        ari_scores.append(ari)
        print(f"\n n_init = {n_init}, Chỉ số Adjusted Rand Index: {ari: .4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(n_init_values, ari_scores, marker='o', color='b', linestyle='-')
    plt.xlabel('Giá trị n_init')
    plt.ylabel('Chỉ số Ajusted Rand Index')
    plt.title('Ảnh hưởng của n_init đén hiệu suất phân cụm k-means')
    plt.grid(True)
    plt.show()


experiment_with_n_init(X_train, y_train)

# buoc 9 In nhan cum cua du lieu huan luyen
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(X_train)
train_clusters_labels = kmeans.labels_

print("\n Nhãn cụm của dữ liệu huần luyện (20 mẫu đầu tiên): ")
print(train_clusters_labels[:20])

print("\n Hiện thị một số chữ số huấn luyện với nhãn cụm dự đoán")
plot_digits(X_train, train_clusters_labels, title="Chữ số huấn luyện với nhãn cụm dự đoán")

# buoc 10 Du doan nhan cum cho du lieu huan luyen bang phuong thuc predict
predicted_train_labels = kmeans.predict(X_train)
print("\n Nhãn cụm dự đoán cho dữ liệu huấn luyện (20 mẫu đầu tiên, dùng predict)")
print(predicted_train_labels[:20])


# buoc 11 Dinh nghia ham print_cluster de hien thi 10 hinh anh tu moi cum
def print_cluster(X, cluster_labels, n_clusters=10, images_per_cluster=10):
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            selected_indices = cluster_indices[:min(images_per_cluster, len(cluster_indices))]
            selected_images = X[selected_indices]
            selected_labels = cluster_labels[selected_indices]
            num_rows = 2
            num_cols = 5
            print(f'\nCụm {cluster} (hiển thị {len(selected_indices)} hình ảnh):')
            plot_digits(selected_images, selected_labels, num_rows, num_cols, title=f'Chữ số trong cụm {cluster}')
        else:
            print(f"\nCụm {cluster} trống")


print("\nHiển thị 10 hình ảnh từ mỗi cụm (dữ liệu huấn luyện):")
print_cluster(X_train, train_clusters_labels)

# buoc 12 Danh gia hieu suat phan cum bang Adjusted Rand Index
test_cluster_labels = kmeans.predict(X_test)
train_ari = adjusted_rand_score(y_train, train_clusters_labels)
test_ari = adjusted_rand_score(y_test, test_cluster_labels)

print('\n Bước 12: Đánh giá hiệu suất phân cụm bằng Adjusted Rand Index')
print(f"\n Chỉ số Adjusted Rand Index cho tập huấn luyện: {train_ari:.4f}")
print(f'\n Chỉ số Adjusted Rand Index cho tập kiểm tra:{test_ari: .4f}')


# buoc 13 Ve luoi (meshgrid) de hien thi vung phan cum trong khong gian 2D
def plot_kmeans_decision_boundary(X, y, kmeans, title="Vùng phân cụm k-means trong không gian 2D"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    h = 0.5
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_original = pca.inverse_transform(mesh_points)
    mesh_labels = kmeans.predict(mesh_points_original)

    plt.figure(figsize=(10, 8))
    plt.contour(xx, yy, mesh_labels.reshape(xx.shape), cmap='tab10', alpha=0.3)

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolors='k', alpha=0.6)
    plt.colorbar(scatter, label='Nhãn cụm')
    plt.xlabel("Thành phần chính 1")
    plt.ylabel("Thành phần chính 2")
    plt.title(title)
    plt.show()


print('\nBước 13: Vẽ vùng phân cụm k-means trong không gian 2D: ')
plot_kmeans_decision_boundary(X_train, train_clusters_labels, kmeans,
                              title="Vùng phân cụm k-means trên dữ liệu huấn luyện")
