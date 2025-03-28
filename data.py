"""Build qm9/dimenet and create .txt files from its features!"""
import os
import tensorflow_datasets as tfds

def txt(config):
    """Creates .txt files from a Tensorflow Dataset!"""
    os.makedirs("data", exist_ok=True)
    dataset = tfds.load(config)
    for split in dataset:
        d = next(iter(dataset[split]))
        features = list(d.keys())

        for f in features:
            with open("data/" + split + "-" + f + ".txt", "w", encoding="utf-8") as txtfile:
                for d in dataset[split]:
                    value = d[f].numpy()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    txtfile.write(str(value) + "\n")

txt("qm9/dimenet")
