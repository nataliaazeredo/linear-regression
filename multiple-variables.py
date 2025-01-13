# dados de treinamento
import numpy as np
import pandas as pd
import time


class GeraDados:

    def __init__(self):

        np.random.seed(42)
        n_samples = 100
        self.area = np.random.uniform(50, 300, n_samples)  # Tamanho da casa
        self.num_quartos = np.random.randint(1, 6, n_samples)  # Número de quartos
        self.idade = np.random.randint(1, 100, n_samples)  # Idade da casa
        self.distancia_centro = np.random.uniform(1, 30, n_samples)  # Distância ao centro da cidade
        self.preco = np.random.uniform(100000, 999999, n_samples)

    def tabela(self):

        dados = pd.DataFrame({
            'Area(m2)': self.area,
            'Num_Quartos': self.num_quartos,
            'Idade': self.idade,
            'Distancia_Centro(km)': self.distancia_centro,
            'Preco(reais)': self.preco
        })

        print('Tabela contendo os 5 primeiros dados')
        print(dados.head())

    def featureScaling(self):
        
        area = (self.area - np.mean(self.area))/ (np.max(self.area)-np.min(self.area))
        num_quartos = (self.num_quartos - np.mean(self.num_quartos)) / (np.max(self.num_quartos)-np.min(self.num_quartos))
        idade = (self.idade - np.mean(self.idade)) / (np.max(self.idade)-np.min(self.idade))
        distancia_centro = (self.distancia_centro - np.mean(self.distancia_centro)) / (np.max(self.distancia_centro)-np.min(self.distancia_centro))

        Y = (np.array(self.preco))
        X = np.array([np.ones(100), area, num_quartos, idade, distancia_centro]).T
        theta = np.zeros(5)

        return X, Y, theta


class Calculadora:

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def MSE(self, theta):
        m = len(self.y)
        Y_pred = np.dot(self.x, theta)  # produto escalar
        MSE = (1/(2*m)) * np.sum((Y_pred - self.y)**2)
        return MSE

    def gradiente(self, theta, l_r, epoch):
        m = len(self.y)
        custo = []

        for i in range(epoch):

            Y_pred = np.dot(self.x, theta)
            erro = Y_pred - self.y
            gradiente = (1/m) * np.dot(self.x.T, erro)
            theta = theta - l_r * gradiente  # atualização do parametro
            custo.append(self.MSE(theta))

        return theta


class Assistente(Calculadora):

    def __init__(self, X, Y):
        super().__init__(X, Y)

    def estimativa(self, theta):

        theta_otimizado = self.gradiente(np.zeros(self.x.shape[1]), l_r=0.001, epoch=800)

        area = float(input('Area da casa em metros quadrados: '))
        quartos = int(input('Números de quartos: '))
        idade = int(input('Idade da casa em anos: '))
        distancia = float(input('Distância do centro da cidade em quilometros: '))

        x = np.array([1, area, quartos, idade, distancia])

        y_usuario = np.dot(theta_otimizado, x.T)
        print(f'O valor estimado da casa é: R$ {y_usuario:.2f}')


def sistema():

    dados = GeraDados()
    X, Y, theta = dados.featureScaling()
    assistente = Assistente(X, Y)

    print('Bem vindo ao programa de estimativas de preços imobiliarios')

    while True:

        time.sleep(0.5)
        print('\nMenu')
        print('1 - Mostrar dados de treinamento')
        print('2 - Fazer previsão')
        print('3 - Sair')

        opcao = int(input('Digite o número da opção desejada: '))

        if opcao == 1:
            dados.tabela()
            time.sleep(0.5)

        elif opcao == 2:
            print('Para realizar a previsão alguns dados serão necessários.')
            time.sleep(0.5)
            assistente.estimativa(theta)
            time.sleep(0.5)

        elif opcao == 3:
            print('Encerrando o sistema')
            break

        else:
            print('Opção inválida')

sistema()