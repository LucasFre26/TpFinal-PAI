```markdown
# Detector de Pneumonia em Raios-X

Um projeto de aprendizado de máquina para classificação de imagens de raios-X torácicos em **Normal** ou **Pneumonia**, utilizando redes neurais convolucionais (CNN) com TensorFlow.

## 📋 Pré-requisitos

- Python 3.8+
- Pip
- Git (opcional)

## ⚙️ Uso

Para configurar e usar o projeto, siga os passos abaixo:

1.  **Crie e ative um ambiente virtual**
    ```bash
    python -m venv venv
    # Linux/Mac:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
    ```

2.  **Instale as dependências:**
    ```bash
    pip install tensorflow opencv-python-headless scikit-learn matplotlib tqdm Pillow
    ```
    (Você pode criar um `requirements.txt` com estas dependências para facilitar futuras instalações).

3.  **Organize os dados:**
    As imagens de raios-X devem ser organizadas nas seguintes pastas dentro do diretório `data/`:

    ```
    data/
    ├── train/
    │   ├── NORMAL/
    │   │   └── img_001.jpeg
    │   │   └── img_002.jpeg
    │   └── PNEUMONIA/
    │       └── img_003.jpeg
    │       └── img_004.jpeg
    └── val/
        ├── NORMAL/
        │   └── img_005.jpeg
        └── PNEUMONIA/
            └── img_006.jpeg
    ```
    - `train/`: Contém as imagens para treinamento.
    - `val/`: Contém as imagens para validação durante o treinamento.
    - `NORMAL/`: Imagens de raios-X normais.
    - `PNEUMONIA/`: Imagens de raios-X com pneumonia.

4.  **Treinar o Modelo:**
    O script `train_model.py` é responsável por carregar e pré-processar os dados, construir e treinar a CNN, e salvar o modelo treinado.
    ```bash
    python train_model.py
    ```
    Durante o treinamento, o melhor modelo será salvo em `models/pneumonia_model.keras`. Logs de treinamento e visualizações do TensorBoard serão gerados na pasta `logs/`.

5.  **Executar a Interface Gráfica (GUI):**
    Após treinar o modelo e ter o arquivo `pneumonia_model.keras` salvo na pasta `models/`, você pode usar a GUI para fazer previsões em novas imagens.
    ```bash
    python gui.py
    ```
    A GUI permitirá que você carregue imagens de raios-X e obtenha uma previsão (NORMAL ou PNEUMONIA) com um nível de confiança.

## 📁 Estrutura do Projeto

```
.
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEMONIA/
├── models/
│   └── pneumonia_model.keras  # Modelo treinado
├── logs/                     # Logs de treinamento e TensorBoard
├── gui.py                    # Interface gráfica do usuário
├── predict.py                # Módulo para fazer previsões
├── preprocess.py             # Módulo para pré-processamento de imagens
└── train_model.py            # Script para treinar o modelo
```

## 🧠 Módulos do Projeto

-   **`preprocess.py`**: Contém a função `load_and_preprocess_images` que carrega imagens de um diretório, aplica redimensionamento com padding, equalização adaptativa de histograma (CLAHE), desfoque Gaussiano, detecção de bordas Canny, normalização e expansão de dimensão.
-   **`train_model.py`**: Define a arquitetura da CNN, compila o modelo com otimizador Adam e métricas relevantes (acurácia, AUC, precisão, recall). Lida com o desbalanceamento de classes e incorpora callbacks para `ModelCheckpoint`, `EarlyStopping`, `CSVLogger` e `TensorBoard`.
-   **`predict.py`**: Contém a classe `PneumoniaPredictor` que carrega o modelo treinado e implementa o pipeline de pré-processamento de imagem (`preprocess_image`) e a função de inferência (`predict`) para classificar novas imagens de raios-X.
-   **`gui.py`**: Implementa a interface gráfica do usuário usando `tkinter`. Permite o upload de imagens, exibe a imagem carregada, mostra o resultado da previsão e um indicador de progresso. Utiliza `threading` para manter a interface responsiva durante o processamento.

## ✨ Pré-processamento de Imagem

O pré-processamento é uma etapa fundamental para o desempenho do modelo. As imagens de raios-X passam pelas seguintes transformações:

1.  **Leitura em escala de cinza**: As imagens são convertidas para escala de cinza para reduzir a complexidade.
2.  **Redimensionamento com Padding**: Imagens são redimensionadas para `(224, 224)` e preenchidas com zeros para manter a proporção original, evitando distorções.
3.  **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Esta técnica melhora o contraste local das imagens, tornando detalhes mais visíveis, especialmente em áreas de baixo contraste.
4.  **Desfoque Gaussiano**: Aplica um filtro para suavizar a imagem e reduzir ruídos, o que pode ajudar a destacar características importantes.
5.  **Canny Edge Detection**: Um algoritmo multi-estágio para detectar uma ampla gama de bordas em imagens. Isso ajuda o modelo a focar em contornos e estruturas.
6.  **Normalização**: Os valores dos pixels são escalados para o intervalo `[0, 1]`, um requisito comum para treinamento de redes neurais.
7.  **Expansão de Dimensão**: Uma dimensão extra é adicionada para representar o canal da imagem (para imagens em escala de cinza, é 1), tornando-a compatível com o formato de entrada esperado pela CNN.
```