"""Build QM9 dataset and write its features to .txt files"""
import tensorflow_datasets as tfds
qm9 = tfds.load("qm9/dimenet")
train = qm9["train"]
validation = qm9["validation"]
test = qm9["test"]
def txt(data, prefix):
    for d in data:
        txt = open(prefix + d + ".txt")
        txt.write(d)
        txt.close()
txt(train, "train-")
txt(validation, "validation-")
txt(test, "test-")
