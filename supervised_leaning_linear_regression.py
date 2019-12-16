from random import randint

from sklearn.linear_model import LinearRegression


# f(x) = 2a + 3b + 5c = y
def linear_function_x(a, b, c):
    return (2 * a) + (3 * b) + (5 * c)


class SupervisedLeaningLinearRegression:
    # Limite de vezes que o modelo sera treinado.
    TRAIN_SET_COUNT = 100
    # Limite de valores para a, b, c.
    TRAIN_SET_LIMIT = 1000

    # Listas representando entradas 'X' e saidas 'Y' da f(x) = y.
    TRAIN_INPUT_X = list()
    TRAIN_OUTPUT_Y = list()

    # Create and append a randomly generated data set to the input and output.
    def generate_input_output(self):
        print('Gerando valores para o treino!')
        for i in range(self.TRAIN_SET_COUNT):
            a = randint(0, self.TRAIN_SET_LIMIT)
            b = randint(0, self.TRAIN_SET_LIMIT)
            c = randint(0, self.TRAIN_SET_LIMIT)

            self.TRAIN_INPUT_X.append([a, b, c])
            self.TRAIN_OUTPUT_Y.append(linear_function_x(a, b, c))

    def train(self):
        # Criar objeto de regressão linear(parâmetro -1 é para utilizar todos os cores do processador).
        predictor = LinearRegression(n_jobs=-1)
        predictor.fit(X=self.TRAIN_INPUT_X, y=self.TRAIN_OUTPUT_Y)  # Adequa o modelo linear (aproximado).

        print('Terminou o treino!')
        return predictor
