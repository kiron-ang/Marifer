print(0)

import tensorflow
import tensorflow_datasets
import matplotlib.pyplot

qm9 = tensorflow_datasets.load("qm9/original")
qm9 = qm9["train"]
qm9 = tensorflow_datasets.as_dataframe(qm9)

for column in qm9.columns:
    try:
        matplotlib.pyplot.hist(qm9[column])
        matplotlib.pyplot.savefig(f"visualizations/hist_{column}.png", dpi = 800)
        print(column, "histogram generated!")

    except Exception as e:
        print(column, "histogram failed:", e)

print(1)