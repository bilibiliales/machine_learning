
import numpy as np
import tensorflow.keras as ka
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

(x_train, y_train), (x_test, y_test) = ka.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train = ka.utils.to_categorical(y_train, 10)
y_test = ka.utils.to_categorical(y_test, 10)

model = ka.Sequential([
    ka.layers.Input(shape=(28, 28, 1)),

    ka.layers.Conv2D(32, (5, 5), activation="relu"),
    ka.layers.MaxPooling2D((2, 2)),

    ka.layers.BatchNormalization(),
    ka.layers.Dropout(0.2),

    ka.layers.Flatten(),
    ka.layers.Dense(64, activation="relu"),
    ka.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率：{test_acc:.4f}")

model.save("mnist.keras")