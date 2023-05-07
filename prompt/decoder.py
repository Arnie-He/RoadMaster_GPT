import tensorflow as tf
from transformer import TransformerBlock, PositionalEncoding
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, function_features, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.function_features = function_features

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(self.hidden_size,activation="leaky_relu")

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(self.vocab_size,self.hidden_size,self.window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(self.hidden_size)

        # Define classification layer (logits)
        self.classifier = tf.keras.layers.Dense(self.vocab_size)

    def call(self, encoded_images, prompts):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        encoded_images = tf.expand_dims(encoded_images,axis=1)
        pos_embedding = self.encoding(prompts)
        image_embeddings = self.image_embedding(encoded_images)
        decoder_output = self.decoder(pos_embedding,image_embeddings)
        logits = self.classifier(decoder_output)
        return logits
    
    def get_config(self):
        config = {
            "vocab_size": self.vocab_size, 
            "hidden_size": self.hidden_size,
            "window_size": self.window_size
        }
        return config