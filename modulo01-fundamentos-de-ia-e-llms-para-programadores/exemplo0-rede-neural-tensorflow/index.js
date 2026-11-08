import * as tf from '@tensorflow/tfjs'


// epochs e onEpochEnd são opcionais (a página HTML usa para escolher as épocas e mostrar o progresso)
export async function trainModel(inputXs, outputYs, { epochs = 100, onEpochEnd } = {}) {
    const model = tf.sequential();

    // ## Primeira camada da rede neural    
    // Entrada de 7 posicoes de dados (idade, 3 cores e 3 localizacoes)
    const in_inputShape = [7];
    
    //80 neuronios por causa da baixa base de treino, quanto mais neuronios mais complexidade 
    // a rede pode aprender e consequentemente vai consumir mais processamento    
    const in_units = 80;

    // a ReLU age como um filtro para deixar somente os dados interessantes seguirem viagem na rede neural
    // se a info chegar no neuronio é positiva passa pra frente se for 0 ou negativo joga fora
    const in_activation = 'relu';

    model.add(tf.layers.dense({inputShape: in_inputShape, units: in_units, activation: in_activation}))

    // ## Segunda camada da rede neural
    // Saida: 3 neuronios por causa das 3 categorias 
    const out_units = 3;

    // a softmax normaliza a saida em probabilidades
    const out_activation = 'softmax';

    model.add(tf.layers.dense({units: out_units, activation: out_activation}));

    // ## Compilando o modelo 
    // Modelo optimizer Adam (Adaptative Moment Estimation)
    // é um treinador pessoal moderno que ajusta os pesos de forma eficiente e inteligente
    // ele aprende com o historico de erros e acertos
    const model_optimizer = 'adam'

    // Compara o que o modelo acha (scores de cada categoria) com a resposta certa
    // categoria sempre vai ser [1,0,0]
    // Quanto mais distante da previsao do modelo da resposta correta maior o erro (loss)
    // modo usado em classificacao de imagens, recomendacao, categoriazacao de usuario
    // qualquer coisa em que a resposta certa é apenas uma entre varias possiveis
    //é o que o modelo tenta minimizar. Ele serve para ajustar os pesos, mas o número é difícil de interpretar.
    const model_loss = 'categoricalCrossentropy';

    // é a porcentagem de previsões em que a categoria com maior score é a correta. 
    // Por exemplo, 0.85 significa que o modelo acertou 85% dos casos. Ela não influencia o aprendizado e serve só para você acompanhar.
    // Essa configuracao só faz o TensorFlow calcular a acurácia a cada época e incluí-la no log e no retorno do model.fit.
    const model_metrics = ['accuracy']

    model.compile({optimizer: model_optimizer , loss:model_loss, metrics: model_metrics});


    // Treinamento do modelo

    await model.fit(inputXs, outputYs,
         {
            verbose: 0,     // controla o que é impresso durante o treino.
            epochs,         // quantas vezes o modelo passa por todo o conjunto de treino. A cada época ele calcula o erro e ajusta os pesos
            shuffle: true,  //  embaralha a ordem dos exemplos a cada época para evitar o VIES.
            callbacks: { onEpochEnd }  // chamado ao final de cada época (epoch, log)
         }
    )

    return model;

}

export async function predict(model, pessoa) {
    // transformar o array js para o tensor js
    const tfInput = tf.tensor2d(pessoa)

    //faz a predicao (output vetor de 3 probabilidades)
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    
    return predArray[0].map((prob, index)=> ({prob, index}))

}


// O exemplo abaixo só roda no Node (npm start). No navegador, o index.html importa apenas
// as funções trainModel e predict acima.
if (typeof window === 'undefined') {

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

inputXs.print();
outputYs.print();

// Quanto mais dado melhor para o algoritmo entender melhor os padroes complexos dos dados
const model = await trainModel(inputXs,outputYs);


// passando uma pessoa fora do treinamento
const pessoa = {nome: 'zé', idade: 28, cor: 'verde', localizacao: 'São Paulo'};
// Normalizando a idade da pessoa -> idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
// idade_min = 25 , idade_max = 40, então (28 -25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2,    // idade normalizada
        1,      // cor azul
        0,      // cor vermelho
        0,      // cor verde
        0,      // sao paulo
        1,      // rio
        0       // curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado);

const results = predictions
                    .sort((a,b) => b.prob - a.prob)
                    .map( p => `${labelsNomes[p.index]} (${(p.prob*100).toFixed(2)})%`)
                    .join(`\n`)

console.log(results)

}

