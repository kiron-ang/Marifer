import tensorflow as tf
import tensorflow_datasets as tfds
qm9 = tfds.load("qm9/dimenet")
for q in qm9:
    print(q)