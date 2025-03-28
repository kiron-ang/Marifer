"""Build QM9 dataset and write its features to .txt files"""
import tensorflow_datasets as tfds
qm9 = tfds.load("qm9/dimenet")
train = qm9["train"]
validation = qm9["validation"]
test = qm9["test"]
def txt(data, prefix):
    features = []
    for d in data:
        for d_ in data:
            features.append(d_)
        break
    for f in features:
        txt = open(prefix + f + ".txt")
        txt.write(f)
        txt.close()
txt(train, "train-")
txt(validation, "validation-")
txt(test, "test-")
