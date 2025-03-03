## PANDAS (LT)
# import pandas as pd

# cau 1
# diameters = [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528]
# ds = pd.Series(diameters)
# print(ds.iloc[0])

# cau 2
# ds = pd.Series(diameters, index=["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"])
# print(ds)

# cau 3
# print(ds["Earth"])

# cau 4
# print(ds["Mercury" : "Mars"])

# cau 5
# print(ds[["Earth", "Jupyter", "Neptune"]])

# cau 6
# ds["Pluto"] = 2370
# print(ds)

# cau 7
# data = {
#     "diameter": [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370],
#     "avg_temp": [167, 464, 15, -65, -110, -140, -195, -200, -225],
#     "gravity": [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
# }
# planets = pd.DataFrame(data, columns=["diameter", "avg_temp", "gravity"])
# planets.index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"]
# print(planets)

# cau 8
# print(planets.head(3))

# cau 9
# print(planets.tail(2))

# cau 10
# print(planets.columns)

# cau 11
# print(planets.index)

# cau 12
# print(planets["gravity"])

# cau 13
# print(planets[["gravity", "diameter"]])

# cau 14
# print(planets.loc["Earth", "gravity"])

# cau 15
# print(planets.loc["Earth", ["diameter", "gravity"]])

# cau 16
# print(planets.loc["Earth" : "Saturn", ["diameter", "gravity"]])

# cau 17
# print(planets[planets["diameter"] > 1000])

# cau 18
# print(planets[planets["diameter"] > 100000])

# cau 19
# print(planets[(planets["avg_temp"] > 0) & (planets["gravity"] < 5)])

# cau 20
# print(planets.sort_values("diameter"))

# cau 21
# print(planets.sort_values("diameter", ascending=False))

# cau 22
# print(planets.sort_values("gravity", ascending=False))

# cau 23
# print(planets.loc["Mercury"].sort_values())

## SEABORN (LT)
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# tips = sns.load_dataset("tips")
# sns.set_style("whitegrid")
# g = sns.lmplot(x="tip",
#                y="total_bill",
#                data=tips,
#                aspect=2)
# g = (g.set_axis_labels("Tip", "Total bill(USD)")).set(xlim=(0,10), ylim=(0,100))
# plt.title("title")
# plt.show()

# ---------------------------------------------------------------------------------------------
## PANDAS (BT)
import pandas as pd

# cau 1
# df = pd.read_csv("04_gap-merged.tsv", sep="\t")
# print(df.head(5))

# cau 2
# print(df.shape)

# cau 3
# print(df.columns)

# cau 4
# print(type(df.columns))

# cau 5
# countries = df["country"]
# print(countries.head(5))

# cau 6
# countries = df["country"]
# print(countries.tail(5))

# cau 7
# data = df[["country", "continent", "year"]]
# print(data.head(5))
# print(data.tail(5))

# cau 8
# print(df.loc[0])
# print(df.loc[99])

# cau 9
# print(df.iloc[:, 0])
# print(df.iloc[:, [0, -1]])

# cau 10
# print(df.loc[df.index[-1]])

# cau 11
# print(df.loc[[0, 9, 99]])
# print(df.iloc[[0, 9, 99]])

# cau 12
# print(df.loc[42, "country"])
# print(df.iloc[42]["country"])

# cau 13
# print(df.iloc[[0, 99, 999], [0, 3, 5]])

# cau 14
# print(df.iloc[:10])

# cau 15
# print(df.groupby("year")["lifeExp"].mean())

# cau 16
# subset = df[["year", "lifeExp"]]
# print(subset.groupby("year").mean())

# cau 17
# data = ["banana", 42]
# ds = pd.Series(data)
# print(ds)

# cau 18
# data = ["Wes MCKinney", "Creator of Pandas"]
# ds = pd.Series(data, index=["Person", "Who"])
# print(ds)

# cau 19
# data = {
#     "Occupation": ["Chemist", "Statistician"],
#     "Born": ["1920-07-25", "1876-06-13"],
#     "Died": ["1958-04-16", "1937-10-16"],
#     "Age": [37, 61]
# }
# df = pd.DataFrame(data, columns=["Occupation", "Born", "Died", "Age"])
# df.index = ["Franklin", "Gosset"]
# print(df)
