import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def train_k_model(ratings, num_users, num_books, n_splits=5):
    X = ratings[['User-ID', 'ISBN']].values
    y = ratings['Book-Rating'].values
    y = pd.to_numeric(y, errors='coerce')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr = 0.001
    batch_s = 64
    ep = 20
    val_mses = []

    # Loop over each fold
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        print(f"Training on fold {fold+1}/{n_splits}...")

        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Remove NaN values
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_val = ~np.isnan(y_val)
        X_val = X_val[valid_val]
        y_val = y_val[valid_val]

        # Define the model
        user_input = Input(shape=(1,))
        user_embedding = Embedding(num_users, 10)(user_input)
        user_flatten = Flatten()(user_embedding)
        user_flatten = Dropout(0.3)(user_flatten)  # Dropout to prevent overfitting

        book_input = Input(shape=(1,))
        book_embedding = Embedding(num_books, 10)(book_input)
        book_flatten = Flatten()(book_embedding)
        book_flatten = Dropout(0.3)(book_flatten)  # Dropout to prevent overfitting

        dot_product = Dot(axes=1)([user_flatten, book_flatten])

        model = Model(inputs=[user_input, book_input], outputs=dot_product)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train the model
        history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=batch_s, epochs=ep, 
                            verbose=1, validation_data=([X_val[:, 0], X_val[:, 1]], y_val), callbacks=[early_stopping])

        # Evaluate the model
        _, mse = model.evaluate([X_val[:, 0], X_val[:, 1]], y_val, verbose=0)
        print(f'Fold {fold+1}, Validation MSE: ', mse)
        val_mses.append(mse)

    avg_mse = np.mean(val_mses)
    print(f'Average Validation MSE across all folds: {avg_mse}')

    # Save the model from the last fold
    model.save(f'model_lr_{lr}_batch_{batch_s}_epoch_{ep}.h5')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 1, 1) # Change subplot grid to 1x1 since we're only plotting loss now
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right') # Change 'Test' to 'Val' for validation set

    plt.tight_layout()
    plt.savefig(f'model_improved_loss_lr_{lr}_batch_{batch_s}_epoch_{ep}.png')
    plt.show()
    return model