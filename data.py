"""Build qm9/dimenet and create .txt files from its features!"""
import tensorflow_datasets as tfds
def txt(dataset):
    """Creates .txt files from a Tensorflow Dataset!"""
    dataset = tfds.load(dataset)
    for split in dataset:
        features = []
        for s in dataset[split]:
            for s_ in s:
                features.append(s_)
            break
        for f in features:
            with open("data/" + split + f + ".txt", "w", encoding = "utf-8") as txtfile:
                txtfile.write("f")
txt("qm9/dimenet")
