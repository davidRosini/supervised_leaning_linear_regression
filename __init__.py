from supervised_leaning_linear_regression import SupervisedLeaningLinearRegression


def main():
    lr = SupervisedLeaningLinearRegression()

    lr.generate_input_output()

    predictor = lr.train()

    predict(predictor)


def predict(predictor):
    end = 0
    while end == 0:
        input_x = input('Entre com os valores de a, b e c: ')
        input_x = [int(x) for x in input_x.split(',')]
        input_x = [input_x]

        # Preve qual o valor de Y usando o dados dos testes do modelo linear.
        outcome = predictor.predict(X=input_x)
        # Estimativa dos coeficientes da função do problema da regressão linear.
        coefficients = predictor.coef_

        print('\nValor de y: {} \n Coeficientes: {}'.format(outcome, coefficients))
        end = input('\nDigite um número diferente de 0 para terminar: ')
        end = int(end)


main()
