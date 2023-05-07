import tensorflow as tf
class Classifier(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])
    
    def call(self, input):
        x = input
        x = self.classifier(x)
        return x