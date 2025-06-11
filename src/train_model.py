import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.utils import class_weight
from preprocess import load_and_preprocess_images
import matplotlib.pyplot as plt
import warnings
import datetime

# Configurações
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 15
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'pneumonia_model.keras')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Suprimir avisos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='keras.src.trainers.data_adapters.py_dataset_adapter')

def plot_history(history):
    """Plota o histórico de treinamento com métricas de treino e validação."""
    metrics = {
        'accuracy': ('Acurácia', 'accuracy', 'val_accuracy'),
        'loss': ('Perda', 'loss', 'val_loss'),
        'auc': ('AUC', 'auc', 'val_auc'),
        'precision': ('Precisão', 'precision', 'val_precision'),
        'recall': ('Recall', 'recall', 'val_recall')
    }
    
    plt.figure(figsize=(20, 12))
    
    for i, (key, (title, metric_name, val_metric_name)) in enumerate(metrics.items(), 1):
        if metric_name in history.history:
            plt.subplot(2, 3, i)
            plt.plot(history.history[metric_name], label='Treino')
            plt.plot(history.history[val_metric_name], label='Validação')
            plt.title(title)
            plt.xlabel('Época')
            plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_callbacks():
    """Cria callbacks para monitoramento do treinamento."""
    callbacks = []
    
    # Salva o melhor modelo
    callbacks.append(
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # Early stopping
    callbacks.append(
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Log para CSV
    callbacks.append(
        CSVLogger(
            os.path.join(LOG_DIR, 'training_log.csv'),
            append=True
        )
    )
    
    # TensorBoard
    log_dir = os.path.join(LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks.append(
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    )
    
    return callbacks

def main():
    # 1. Carregar e pré-processar os dados
    X_train, y_train, class_names = load_and_preprocess_images(TRAIN_DIR, IMG_SIZE)
    X_val, y_val, _ = load_and_preprocess_images(VAL_DIR, IMG_SIZE)

    print(f"\nDados carregados:")
    print(f"- Treino: {X_train.shape[0]} imagens")
    print(f"- Validação: {X_val.shape[0]} imagens")
    print(f"Classes: {class_names}\n")

    # 2. Calcular pesos das classes
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(y_train),
                                                    y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Pesos das classes: {class_weights_dict}\n")

    # 3. Criar datasets
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # 4. Criar modelo
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # 5. Compilar modelo com mais métricas
    model.compile(
        optimizer=Adam(),
        loss=BinaryCrossentropy(),
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    # Resumo do modelo
    model.summary()

    # 6. Criar callbacks
    callbacks = create_callbacks()

    # 7. Treinar modelo
    print("\nIniciando treinamento...\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    # 8. Avaliação final
    print("\nAvaliação final no conjunto de validação:")
    val_results = model.evaluate(val_ds, verbose=0)
    print(f"- Loss: {val_results[0]:.4f}")
    print(f"- Accuracy: {val_results[1]:.4f}")
    print(f"- AUC: {val_results[2]:.4f}")
    print(f"- Precision: {val_results[3]:.4f}")
    print(f"- Recall: {val_results[4]:.4f}")

    # 9. Plotar histórico
    plot_history(history)

if __name__ == '__main__':
    # Criar diretórios se não existirem
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    main()