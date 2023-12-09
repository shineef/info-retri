import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, MultiHeadAttention, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D
from keras.models import Model
from keras import Input

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def transformer_block(embed_dim, num_heads, ff_dim, rate=0.1, name=None):
    input_layer = Input(shape=(None, embed_dim))

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name=f"{name}_mha")(input_layer, input_layer)

    # attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(input_layer, input_layer)
    attention_output = Dropout(rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(input_layer + attention_output)
    
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(embed_dim)(ff_output)
    ff_output = Dropout(rate)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    return Model(inputs=input_layer, outputs=ff_output, name=name)

def build_transformer_model(user_size, book_size, embed_dim, num_heads, ff_dim, num_blocks, rate=0.005):
    # Input layer: users and books
    user_input = Input(shape=(1,))
    book_input = Input(shape=(1,))

    # embedding layer
    user_embedding = Embedding(user_size, embed_dim)(user_input)
    book_embedding = Embedding(book_size, embed_dim)(book_input)

    # Merge embedding layers
    combined = keras.layers.Concatenate()([user_embedding, book_embedding])
    combined = keras.layers.Reshape((1, -1))(combined)  # Adjust to 3D shape

    x = combined
    # for _ in range(num_blocks):
    #     transformer_block_model = transformer_block(embed_dim * 2, num_heads, ff_dim, rate)
    #     x = transformer_block_model(x)
    for i in range(num_blocks):
        transformer_block_model = transformer_block(embed_dim * 2, num_heads, ff_dim, rate, name=f"transformer_block_{i}")
        x = transformer_block_model(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(rate)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[user_input, book_input], outputs=outputs)
    return model


# Example usage
maxlen = 100  # Length of input sequences
vocab_size = 10000  # Size of vocabulary
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
num_blocks = 1  # Number of transformer blocks

transformer_model = build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks)
transformer_model.summary()