from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

coordinates = [
    (50, 127),
    (60, 253),
    (70, 1009),
    (80, 16129),
]

plot = sns.lineplot(
    x=[x for x, _ in coordinates],
    y=[y for _, y in coordinates],
)
plot.set(yscale='log')
plt.show()