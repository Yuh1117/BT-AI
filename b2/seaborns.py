import matplotlib.pyplot as plt
import seaborn as sns

# cau 2
# print(sns.get_dataset_names())

# cau 3
tips = sns.load_dataset("tips")
# print(tips.head())

# cau 4
# sns.set_style("whitegrid")
# g = sns.lmplot(x="total_bill",
#                y="tip",
#                data=tips,
#                aspect=2)
# g = (g.set_axis_labels("Total bill(USD)", "Tip")).set(xlim=(0,100), ylim=(0,10))
# plt.title("Total Bill vs Tip")
# plt.show()

# cau 5
# sns.set_theme(font_scale=1.2)
# sns.set_style("darkgrid")
# g = sns.lmplot(x="total_bill",
#                y="tip",
#                data=tips,
#                aspect=2)
# g = (g.set_axis_labels("Total bill(USD)", "Tip")).set(xlim=(0, 100), ylim=(0, 10))
# plt.title("Total Bill vs Tip")
# plt.show()

# cau 6
# sns.set_theme(font_scale=1.2)
# sns.set_style("darkgrid")
# g = sns.lmplot(x="total_bill",
#                y="tip",
#                data=tips,
#                aspect=2,
#                hue="day")
# g = (g.set_axis_labels("Total bill(USD)", "Tip")).set(xlim=(0,100), ylim=(0,10))
# plt.title("Total Bill vs Tip by Day")
# plt.show()

# cau 7
# sns.set_theme(font_scale=1.2)
# sns.set_style("darkgrid")
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x="total_bill",
#                 y="tip",
#                 data=tips,
#                 hue="day",
#                 size="size")
#
# plt.xlabel("Total bill(USD)")
# plt.ylabel("Tip")
# plt.xlim(0, 100)
# plt.ylim(0, 10)
# plt.title("Total Bill vs Tip by Day and Size")
# plt.show()

# cau 8
# sns.set_theme(font_scale=1.2)
# sns.set_style("darkgrid")
# g = sns.lmplot(x="total_bill",
#                y="tip",
#                data=tips,
#                aspect=1.5,
#                hue="day",
#                col="time")
# g = (g.set_axis_labels("Total bill(USD)", "Tip")).set(xlim=(0,100), ylim=(0,10))
# g.set_axis_labels("Total Bill (USD)", "Tip")
# plt.show()

# cau 9
sns.set_theme(font_scale=1.2)
sns.set_style("darkgrid")
g = sns.lmplot(x="total_bill",
               y="tip",
               data=tips,
               aspect=1.5,
               hue="day",
               col="time",
               row="sex")
g = (g.set_axis_labels("Total bill(USD)", "Tip")).set(xlim=(0,100), ylim=(0,10))
g.set_axis_labels("Total Bill (USD)", "Tip")
plt.show()