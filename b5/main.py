import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# fig, ax = plt.subplots(figsize=(8, 6))

# num_nodes_per_layer = 10

# G1 = nx.Graph()
# G2 = nx.Graph()

# central_node_1 = "C1"
# central_node_2 = "C2"
# G1.add_node(central_node_1)
# G2.add_node(central_node_2)

# for i in range(num_nodes_per_layer):
#     G1.add_node(f"L1_{i}")
#     G2.add_node(f"L2_{i}")

# weights1 = np.linspace(0.025, 0.200, num_nodes_per_layer)
# weights2 = np.linspace(0.025, 0.200, num_nodes_per_layer)

# for i in range(num_nodes_per_layer):
#     G1.add_edge(central_node_1, f"L1_{i}", weight=weights1[i])
#     G2.add_edge(central_node_2, f"L2_{i}", weight=weights2[i])

# pos1 = nx.circular_layout(G1)
# pos2 = nx.circular_layout(G2)

# for node in pos2:
#     pos2[node][1] -= 3

# pos = {**pos1, **pos2}

# node_color1 = ['yellow'] + [
#     'red' if abs(weights1[i] - 0.200) < 0.01 else 'orange' if abs(weights1[i] - 0.150) < 0.01 else 'brown' for i in
#     range(num_nodes_per_layer)]
# node_color2 = ['yellow'] + ['red' if abs(weights1[i] - 0.200) < 0.01 else 'brown' for i in range(num_nodes_per_layer)]

# nx.draw_networkx_nodes(G1, pos, node_color=node_color1, node_size=500, ax=ax)
# nx.draw_networkx_nodes(G2, pos, node_color=node_color2, node_size=500, ax=ax)

# edge_color1 = [G1[u][v]["weight"] for u, v in G1.edges()]
# edge_color2 = [G2[u][v]["weight"] for u, v in G2.edges()]
# nx.draw_networkx_edges(G1, pos, edge_color=edge_color1, edge_cmap=plt.cm.Reds, ax=ax)
# nx.draw_networkx_edges(G2, pos, edge_color=edge_color2, edge_cmap=plt.cm.Reds, ax=ax)

# ax.text(-1.5, 0, 'layer: i = 1', fontsize=12, verticalalignment='center')
# ax.text(-1.5, -2, 'layer: i = 2', fontsize=12, verticalalignment='center')

# sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0.025, vmax=0.200))
# plt.colorbar(sm, ax=ax, label='')

# plt.title("Mutilayer Network")
# plt.axis('off')
# plt.show()



# Tạo đồ thị multilayer
G = nx.Graph()

n_nodes = 10

positions = {}
colors = []

# Layer 1 (vòng tròn)
for i in range(n_nodes):
    angle = 2 * np.pi * i / n_nodes
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle)
    node = f"1-{i}"
    positions[node] = (x, y)
    G.add_node(node)
    weight = 0.025 + (i / (n_nodes - 1)) * (0.2 - 0.025)
    colors.append(weight)

# Tính tâm hình tròn layer 1
x_vals = [positions[f"1-{i}"][0] for i in range(n_nodes)]
y_vals = [positions[f"1-{i}"][1] for i in range(n_nodes)]
x_center = sum(x_vals) / n_nodes
y_center = sum(y_vals) / n_nodes

# Thêm node nằm ở giữa hình tròn
extra_node = "extra-center"
positions[extra_node] = (x_center, y_center)
G.add_node(extra_node)
colors.append(0.2)  # màu vàng chẳng hạn

# Nối node này với 1 node bất kỳ (ví dụ node 1-2) hoặc nhiều node tùy bạn
G.add_edge("1-5", extra_node)

# Layer 2 (hoa thị)
center_node = "2-center"
G.add_node(center_node)
positions[center_node] = (0, -5)
colors.append(0.2)  # Trung tâm vàng

for i in range(n_nodes):
    angle = 2 * np.pi * i / n_nodes
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle) - 5
    node = f"2-{i}"
    positions[node] = (x, y)
    G.add_node(node)
    weight = 0.025 + (i / (n_nodes - 1)) * (0.2 - 0.025)
    colors.append(weight)
    G.add_edge(center_node, node)

# Kết nối layer 1 với layer 2 bằng các cạnh thẳng đứng
for i in range(n_nodes):
    G.add_edge(f"1-{i}", f"2-{i}")

# Kết nối các node layer 1 thành vòng tròn thiếu 1 cạnh (giả sử bỏ cạnh giữa 1-4 và 1-5)
for i in range(n_nodes):
    if i != 5:
        G.add_edge(f"1-{i}", f"1-{(i + 1) % n_nodes}")

# Vẽ đồ thị
plt.figure(figsize=(6, 6))
nodes = nx.draw_networkx_nodes(G, positions, node_color=colors, cmap='inferno', node_size=300)

# Vẽ các cạnh layer 1 (vòng tròn + extra node)
for u, v in G.edges():
    if (u.startswith("1-") and v.startswith("1-")) or (u == "extra-center" and v.startswith("1-")):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='black', width=7)
    elif u.startswith("1-") and v.startswith("2-"):
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='gray', alpha=0.3, width=10)
    else:
        nx.draw_networkx_edges(G, positions, edgelist=[(u, v)], edge_color='black', width=7)

plt.colorbar(nodes, shrink=0.7, label='Weight')
plt.text(-4.5, 3.5, 'layer $i = 1$', fontsize=12, rotation=90, va='center')
plt.text(-4.5, -3.5, 'layer $i = 2$', fontsize=12, rotation=90, va='center')
plt.axis('off')
plt.tight_layout()
plt.show()