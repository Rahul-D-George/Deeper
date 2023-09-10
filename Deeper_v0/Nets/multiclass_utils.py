import tensorflow as tf


class ConvClassifier:
    def __init__(self, X, Ydata, epochs=None, summarise=False):
        self.nclasses = len(Ydata)

        self.X = X
        self.Y = tf.keras.utils.to_categorical(Ydata, num_classes=self.nclasses)
        if epochs is None:
            self.epochs = 10
        else:
            self.epochs = epochs

        self.model = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=(1, 1), input_shape=(185, 185, 3)),
            tf.keras.layers.Conv2D(16, 7, strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 5, strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.nclasses, activation="softmax"),
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        if summarise: print(self.model.summary())

    def train(self):
        self.model.fit(self.X, self.Y, epochs=self.epochs)
