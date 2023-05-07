import tensorflow as tf
class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.vocab_size = 1000
        self.preprocess = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size)
        self.encoder = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=len(self.preprocess.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    ])

    def call(self, input):
        x = input
        x = self.preprocess(x)
        x = self.encoder(x)
        return x