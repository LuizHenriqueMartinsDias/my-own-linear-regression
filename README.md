# My Own Linear Regression

Implementação de Regressão Linear "do zero" (sem `scikit-learn`), com duas abordagens:

- **Mínimos Quadrados Ordinários (OLS)** — solução fechada, para regressão simples (uma variável).
- **Gradiente Descendente (GD)** — treinamento iterativo, com suporte a múltiplas variáveis (regressão múltipla).

O objetivo do projeto é didático: entender e implementar manualmente a matemática por trás da regressão linear (soma de quadrados, gradientes, padronização, R²) em vez de usar bibliotecas prontas.

## Estrutura do projeto

```
my-own-linear-regression/
├── data/
│   └── toy_dataset          # CSV usado para treino/validação
├── src/
│   └── leastsquares.py      # Implementação das classes e do pipeline
└── README.md
```

## Dataset

O arquivo `data/toy_dataset` contém dados sintéticos de imóveis, com as colunas:

| coluna        | descrição                          |
|----------------|-------------------------------------|
| `area_m2`      | área do imóvel em m²                |
| `quartos`      | número de quartos                   |
| `idade_anos`   | idade do imóvel em anos             |
| `y`            | valor alvo (preço) a ser previsto   |

## Classes

### `LinearRegression`

Regressão linear simples (uma variável `x`), resolvida via fórmula fechada de mínimos quadrados:

```python
lr = LinearRegression()
lr.fit(train_dataset)          # DataFrame com colunas "x" e "y"
lr.predict(x)
lr.score(validation_dataset)   # imprime R²
lr.plot_validation(train_dataset, validation_dataset)  # plota treino, validação e reta ajustada
```

### `LinearRegressionGD`

Regressão linear múltipla, treinada via gradiente descendente (batch), com MSE como função de custo.

```python
lrgd = LinearRegressionGD(learning_rate=0.1, epochs=250)
lrgd.fit(X_train, y_train)     # X_train: DataFrame com as features; y_train: Series
lrgd.predict(X)                # aceita DataFrame ou lista/array
lrgd.score(X, y)                # imprime R²
```

Durante o treino, a cada 10 épocas são impressos MSE, pesos (`coef_`) e viés (`intercept_`) para acompanhar a convergência.

> **Importante:** o modelo é treinado sobre features **padronizadas** (ver seção abaixo). Qualquer previsão sobre dados novos precisa passar pela mesma padronização (mesmos `mean`/`std` do treino) antes de ser passada para `predict`, ou o resultado será inconsistente.

## Funções auxiliares

- **`dataset_split(dataset, split)`** — separa o dataset em treino/validação de forma aleatória (com `random_state=42` para reprodutibilidade), retornando `X_train, y_train, X_validation, y_validation`.
- **`standarization(x, mean=None, std=None)`** — padroniza os dados (`(x - mean) / std`). Se `mean`/`std` não forem informados, são calculados a partir de `x`; isso permite reaproveitar as estatísticas do treino ao padronizar o conjunto de validação ou novas amostras.

Por que padronizar? O gradiente descendente aqui usa uma única `learning_rate` para todas as features. Como `area_m2`, `quartos` e `idade_anos` estão em escalas muito diferentes, sem padronização os gradientes de cada peso teriam magnitudes muito distintas — tornando difícil (ou impossível) escolher uma taxa de aprendizado que funcione bem para todos os pesos ao mesmo tempo.

## Como rodar

```bash
python src/leastsquares.py
```

Isso vai:
1. Carregar `data/toy_dataset`;
2. Separar em treino (60%) e validação (40%);
3. Padronizar as features;
4. Treinar `LinearRegressionGD`;
5. Imprimir métricas de treino a cada 10 épocas e o R² final na validação.

## Requisitos

- Python 3.10+
- `numpy`
- `pandas`
- `matplotlib`

```bash
pip install numpy pandas matplotlib
```

## Possíveis melhorias futuras

- Guardar `mean`/`std` como atributos do modelo (`self.mean_`, `self.std_`) para padronizar automaticamente dentro de `predict`, evitando o erro de esquecer de padronizar novas amostras.
- Adicionar early stopping / critério de convergência baseado na variação do MSE.
- Suporte a mini-batch ou GD estocástico.
- Testes automatizados comparando os resultados com `scikit-learn`.
