import tensorflow as tf
import numpy as np
import itertools
import time
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregar dataset MNIST
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data, test_data = train_data / 255.0, test_data / 255.0
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# Definir hiperparâmetros para testar
models = ['a', 'b']
learning_rates = [0.01, 0.005, 0.001]
batch_sizes = [25, 50, 100]
l2_norm_clips = [0.5, 1.0, 1.5]
noise_multipliers = [0.5, 1.0, 1.5]
num_microbatches_options = [5, 10]
epochs = 30
delta = 1e-5

# Criar arquivo de resultados
file_path = "resultados.txt"
with open(file_path, "w") as f:
    f.write("Resultados dos testes de modelos DP e não DP\n")
    f.write("Todos os testes foram executados com: delta = {delta} e epochs = {epochs}\n\n")

# Função para gerar batches de tamanho fixo com Poisson Sampling
def poisson_batch_generator(data, labels, batch_size, steps_per_epoch):
    """
    Gera batches de tamanho fixo com amostragem Poisson.
    """
    N = len(data)
    q = batch_size / N
    for _ in range(steps_per_epoch):  # Limitar o número de batches por época
        mask = np.random.rand(N) < q
        X_batch, y_batch = data[mask], labels[mask]
        
        # Preencher ou truncar o batch para o tamanho fixo
        if len(X_batch) > batch_size:
            X_batch, y_batch = X_batch[:batch_size], y_batch[:batch_size]
        elif len(X_batch) < batch_size:
            pad_size = batch_size - len(X_batch)
            X_batch = np.vstack((X_batch, np.zeros((pad_size, 28, 28, 1), dtype=np.float32)))
            y_batch = np.vstack((y_batch, np.zeros((pad_size, 10), dtype=np.float32)))
        
        yield X_batch, y_batch

# Criar arquiteturas
def create_model_a():
    model = Sequential([
        Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(2, 1),
        Conv2D(32, 4, strides=2, padding='valid', activation='relu'),
        MaxPool2D(2, 1),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10)
    ])
    return model

def create_model_b():
    model = Sequential([
        Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, 3, activation='relu'),
        BatchNormalization(),
        Conv2D(32, 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Conv2D(64, 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, 3, activation='relu'),
        BatchNormalization(),
        Conv2D(64, 5, strides=2, padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    return model

test_number = 0
# Gerar todas as combinações de hiperparâmetros
for model, lr, batch_size, l2_norm_clip, noise_multiplier in itertools.product(
    models, learning_rates, batch_sizes, l2_norm_clips, noise_multipliers):

    for dp in [False, True]:  # Testar modelos com e sem DP
        for num_microbatches in (num_microbatches_options if dp else [None]):

            # Ajustar microbatches apenas para DP
            if dp and batch_size % num_microbatches != 0:
                continue  

            # Inicializar o modelo
            model = locals()[f"create_model_{model}"]()

            # Definir função de perda
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE if dp else tf.losses.Reduction.AUTO)

            # Escolher otimizador
            if dp:
                optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
                    l2_norm_clip=l2_norm_clip,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=num_microbatches,
                    learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            if dp:
                steps_per_epoch = len(train_data) // batch_size  # Definir o número de batches por época
                train_dataset = tf.data.Dataset.from_generator(
                    lambda: poisson_batch_generator(train_data, train_labels, batch_size, steps_per_epoch),
                    output_signature=(
                        tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32)
                    )
                ).prefetch(tf.data.experimental.AUTOTUNE)

                val_dataset = tf.data.Dataset.from_generator(
                    lambda: poisson_batch_generator(test_data, test_labels, batch_size, steps_per_epoch),
                    output_signature=(
                        tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32)
                    )
                ).prefetch(tf.data.experimental.AUTOTUNE)

                start_time = time.time()
                history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1)
                end_time = time.time()
            else:
                start_time = time.time()
                history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels), verbose=1)
                end_time = time.time()

            # Avaliação do modelo
            predictions = model.predict(test_data)
            predicted_labels = np.argmax(predictions, axis=1)
            true_labels = np.argmax(test_labels, axis=1)

            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='weighted')
            recall = recall_score(true_labels, predicted_labels, average='weighted')
            f1 = f1_score(true_labels, predicted_labels, average='weighted')

            if dp:
                privacy_statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
                    number_of_examples=train_data.shape[0],
                    batch_size=batch_size,
                    num_epochs=epochs,
                    noise_multiplier=noise_multiplier,
                    delta=delta
                )
            else:
                privacy_statement = "Sem privacidade diferencial"

            with open(file_path, "a") as f:
                f.write(f"---------------------------------------------------------------------------------------------------------------------------------- \n")
                f.write(f"Teste {test_number}: \n\n")
                f.write(f"DP: {dp} | LR: {lr} | Batch: {batch_size} | L2 Clip: {l2_norm_clip} | Noise: {noise_multiplier} | Microbatches: {num_microbatches}\n")
                f.write(f"Acurácia: {accuracy:.4f} | Precisão: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} \n")
                f.write(f"Tempo de Treinamento: {end_time - start_time:.2f} seg\n")
                f.write(f"Privacidade: {privacy_statement}\n\n")
            
            test_number += 1

print("Testes concluídos! Resultados salvos em '{file_path}'.")
