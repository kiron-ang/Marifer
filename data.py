"""Build QM9 dataset and write its features to .txt files"""
import tensorflow as tf
import tensorflow_datasets as tfds
qm9 = tfds.load("qm9/dimenet")
train = qm9["train"]
validation = qm9["validation"]
test = qm9["test"]
for t in train:
    print(t)
