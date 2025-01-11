print('LINEAR REGRESSION WITH ONE VARIABLE')


class Calculadora:

    def __init__(self, x, y):

        self.x = x
        self.y = y

    # Se os dados de treinamento forem poucos, essa função é suficiente.
    def regressao(self):

        sum_x = sum(self.x)
        sum_y = sum(self.y)
        sum_x2 = sum([i**2 for i in self.x])
        sum_xy = sum([self.x[i]*self.y[i] for i in range(len(self.x))])

        n = len(self.x)

        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_x*sum_xy-sum_y*sum_x2)/((sum_x)**2-n*sum_x2)

        return a, b

    def step(self, a, b, l_r):  # Atualização dos parametros

        erro_b = 0
        erro_a = 0
        m = len(self.x)
        def y_custo(x, a, b): return a*x+b
        for i in range(len(self.x)):

            erro_b += y_custo(self.x[i], a, b)-self.y[i]
            erro_a += (y_custo(self.x[i], a, b)-self.y[i])*self.x[i]

        b2 = b-l_r*(1/m)*erro_b
        a2 = a-l_r*(1/m)*erro_a

        return a2, b2

    def MSE(self, a, b):
        custo = 0
        def y_custo(x, a, b): return a*x+b  # Retorna só um valor de y previsto
        for i in range(len(self.x)):
            custo += (y_custo(self.x[i], a, b)-self.y[i])**2

        return custo/(2*len(self.x))

    def previsao(self, a, b):
        y_pred = [a * self.x[i] + b for i in range(len(self.x))]

        return y_pred
    
    # seria melhor se eu tivesse usado precisão
    def gradiente(self, a, b, l_r, epoch):  
        custo = []
        for i in range(epoch):
            a, b = self.step(a, b, l_r)
            custo.append(self.MSE(a, b))

        return a, b


class Assistente(Calculadora):

    def __init__(self):
        super().__init__(x, y)

    def estimativa(self):
        if len(self.x) <= 10:
            a, b = self.regressao()
        else:
            a, b = self.gradiente(0, 0, 0.001, 800)
        x_usuario = float(
            input('Informa o valor de x que deseja fazer a previsão: '))
        y_usuario = a*x_usuario+b
        print(f'O valor previsto é {y_usuario}')


def sistema():

    global x, y

    while True:

        print('MENU')
        print('1 - Realizar uma previsão')
        print('2 - Adicionar dados de treinamento')
        print('3 - Reiniciar dados de treinamento')
        print('4 - Sair')

        escolha = int(input('Qual operação deseja realizar: '))

        assistente = Assistente()
        if escolha == 1:
            assistente.estimativa()

        elif escolha == 2:
            n = int(input("Quantos valores deseja adicionar? "))
            for i in range(n):
                x_novo = float(input('x:'))
                y_novo = float(input('y:'))
                x.append(x_novo)
                y.append(y_novo)
            print(f'O valores atuais são:\n x = {x}\n y = {y}')

        elif escolha == 3:
            x.clear()
            y.clear()
            n = int(input("Quantos novos valores deseja registrar? "))
            for i in range(n):
                x_novo = float(input('x:'))
                y_novo = float(input('y:'))
                x.append(x_novo)
                y.append(y_novo)
            print(f'O valores atuais são:\n x = {x}\n y = {y}')

        elif escolha == 4:
            print('Programa encerrado.')
            break

        else:
            print('Opção inválida. Digite um valor disponível.')


x = [1, 2, 3, 4, 5]
y = [1.3, 1.8, 3.5, 4, 4.6]

sistema()