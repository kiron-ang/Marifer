"""Build QM9 dataset and write its features to .txt files"""
import tensorflow_datasets as tfds
qm9 = tfds.load("qm9/dimenet")
train = qm9["train"]
validation = qm9["validation"]
test = qm9["test"]
def txt(data, prefix):
    """Writes data to .txt files"""
    features = []
    for d in data:
        for d_ in d:
            features.append(d_)
        break
    for f in features:
        with open("data/" + prefix + f + ".txt") as txtfile:
            txtfile.write("f")
txt(train, "train-")
txt(validation, "validation-")
txt(test, "test-")
