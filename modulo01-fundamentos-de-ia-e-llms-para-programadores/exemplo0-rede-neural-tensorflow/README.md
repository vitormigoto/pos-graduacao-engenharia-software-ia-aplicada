# Exemplo 0 - Rede neural com TensorFlow.js

Rede neural simples que aprende a classificar pessoas (idade, cor e localização) em três categorias: `premium`, `medium` e `basic`.

O projeto pode ser executado de duas formas:

- **No terminal (Node.js)**: treina com 3 pessoas fixas e imprime a previsão para uma pessoa nova.
- **No navegador**: uma página onde você edita os dados de treino, treina o modelo e prevê a categoria de uma pessoa.

## Arquivos

| Arquivo | Para que serve |
| --- | --- |
| `index.js` | Cria, treina e usa a rede neural (`trainModel` e `predict`). Também roda o exemplo no Node. |
| `index.html` | Página que importa `trainModel` e `predict` do `index.js`. |
| `package.json` | Dependências e script `start`. |

## Pré-requisitos

- [Node.js](https://nodejs.org) 18 ou superior (testado com a 20).
- Navegador moderno (Chrome, Edge ou Firefox) e conexão com a internet, pois a página carrega o TensorFlow.js por CDN.

## Instalação

Dentro desta pasta:

```bash
npm install
```

> Este projeto usa o pacote `@tensorflow/tfjs` (JavaScript puro). Não use o `@tensorflow/tfjs-node`: o binário pré-compilado dele para Windows não é mais baixado (erro 404) e a instalação cai numa compilação que exige o Visual Studio.
>
> Se a pasta `node_modules` não tiver o pacote, instale com `npm install @tensorflow/tfjs`.

## Rodando no terminal (Node.js)

```bash
npm start
```

O script usa `node --watch`, então reexecuta sozinho sempre que você salvar o `index.js`. Para parar, use `Ctrl + C`.

Saída esperada (os percentuais mudam a cada execução, porque os pesos iniciais são aleatórios):

```
premium (45.91)%
medium (42.89)%
basic (11.20)%
```

O aviso `Hi, looks like you are running TensorFlow.js in Node.js...` é normal e pode ser ignorado.

## Rodando no navegador

**Não abra o `index.html` com dois cliques.** O navegador bloqueia o `import './index.js'` em páginas abertas como `file://`. É preciso servir a pasta por HTTP, de uma das formas abaixo.

**Opção A: Live Server (VS Code)**

1. Instale a extensão *Live Server*.
2. Clique com o botão direito em `index.html` e escolha **Open with Live Server**.

**Opção B: terminal**

```bash
npx serve .
```

Abra o endereço mostrado (normalmente `http://localhost:3000`).

### Como usar a página

1. **Dados de treino**: a tabela já vem com Erick, Ana e Carlos. Edite os campos, remova pessoas com o ✕ ou adicione outras com **+ Adicionar pessoa**. Cada pessoa precisa de uma categoria, que é a resposta certa.
2. Ajuste as **épocas** (quantas vezes o modelo percorre todos os dados) e clique em **Treinar modelo**. O status mostra o progresso, a loss e a acurácia.
3. **Pessoa para prever**: escolha idade, cor e localização e clique em **Prever**. As barras mostram a probabilidade de cada categoria.

Sempre que mudar os dados de treino, treine de novo antes de prever.

## Como funciona

- **Entrada (7 valores):** `[idade normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]`.
  - A idade é normalizada para 0 a 1: `(idade - idade_min) / (idade_max - idade_min)`.
  - Cor e localização usam one-hot encoding (um `1` na posição escolhida e `0` nas outras).
- **Rede:** camada densa com 80 neurônios (`relu`) e camada de saída com 3 neurônios (`softmax`).
- **Saída:** probabilidades das categorias `[premium, medium, basic]`, que somam 100%.
- **Treino:** otimizador `adam`, perda `categoricalCrossentropy` e métrica `accuracy`.

## Problemas comuns

| Sintoma | Causa e solução |
| --- | --- |
| `npm install @tensorflow/tfjs-node` falha com 404 ou "Could not find any Visual Studio" | Use `@tensorflow/tfjs`, como descrito acima. |
| Página em branco ou erro de CORS no console | O `index.html` foi aberto por `file://`. Use o Live Server ou `npx serve .`. |
| `Tensor must have a shape comprised of positive integers` | `units` precisa ser um número (`3`), não um array (`[3]`). |
| Resultados diferentes a cada treino | Normal: os pesos começam aleatórios. Com poucos dados, o resultado varia bastante. Aumente as épocas ou adicione mais pessoas. |
