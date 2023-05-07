import tensorflow as tf
class CNN(tf.keras.Model):
    def __init__(self):
        self.cnn = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        ])
        self.attention = tf.keras.layers.Attention()
        self.layer_norm = tf.keras.layers.LayerNormalization()


    def call(self, input):
        x = input
        x = self.cnn(x)
        x = self.attention(x,x,x)
        x = self.layer_norm(x)
        return x