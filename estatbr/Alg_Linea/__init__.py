class AlgebraLinear:
    # Função para multiplicar uma matriz por um escalar
    @staticmethod
    def multiplicar_matriz_por_escalar(matriz, escalar):
        resultado = []
        for linha in matriz:
            nova_linha = [elemento * escalar for elemento in linha]
            resultado.append(nova_linha)
        return resultado

    # Função para dividir uma matriz por um escalar
    @staticmethod
    def dividir_matriz_por_escalar(matriz, escalar):
        if escalar == 0:
            raise ValueError("Não é possível dividir uma matriz por zero.")
        return AlgebraLinear.multiplicar_matriz_por_escalar(matriz, 1 / escalar)

    # Função para multiplicar uma matriz por um vetor
    @staticmethod
    def multiplicar_matriz_por_vetor(matriz, vetor):
        if len(matriz[0]) != len(vetor):
            raise ValueError("O número de colunas da matriz deve ser igual ao tamanho do vetor.")
        resultado = []
        for linha in matriz:
            soma = sum(elemento * vetor[i] for i, elemento in enumerate(linha))
            resultado.append(soma)
        return resultado


    # Função para resolver um sistema de equações lineares utilizando matrizes
    @staticmethod
    def resolver_sistema_linear(coeficientes, constantes):
        import copy
        matriz_coeficientes = copy.deepcopy(coeficientes)
        matriz_constantes = [[float(constante)] for constante in constantes]
        inversa_coeficientes = AlgebraLinear.matriz_inversa(matriz_coeficientes)

        # Multiplicar a matriz inversa pelos vetores constantes corretamente
        solucao = AlgebraLinear.multiplicar_matriz(inversa_coeficientes, matriz_constantes)

        return [f"{item[0]:.2f}" for item in solucao]


    # Função para calcular a matriz inversa
    @staticmethod
    def matriz_inversa(matriz):
        n = len(matriz)
        identidade = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        matriz_aumentada = [linha + identidade[i] for i, linha in enumerate(matriz)]
        for i in range(n):
            pivo = matriz_aumentada[i][i]
            if pivo == 0:
                raise ValueError("A matriz não possui inversa.")
            for j in range(i, n * 2):
                matriz_aumentada[i][j] /= pivo
            for k in range(n):
                if k != i:
                    fator = matriz_aumentada[k][i]
                    for j in range(i, n * 2):
                        matriz_aumentada[k][j] -= fator * matriz_aumentada[i][j]
        matriz_inversa = [linha[n:] for linha in matriz_aumentada]
        return matriz_inversa

    # Função para elevar uma matriz a uma potência inteira
    @staticmethod
    def potencia_matriz(matriz, potencia):
        if len(matriz) != len(matriz[0]):
            raise ValueError("A matriz deve ser quadrada para ser elevada a uma potência.")
        if potencia == 0:
            n = len(matriz)
            return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        if potencia < 0:
            matriz = AlgebraLinear.matriz_inversa(matriz)
            potencia *= -1
        resultado = matriz
        for _ in range(potencia - 1):
            resultado = AlgebraLinear.multiplicar_matriz(resultado, matriz)
        return resultado


    def decomposicao_LU(matriz):
        from mpmath import mp
        if len(matriz) != len(matriz[0]):
            raise ValueError("A matriz deve ser quadrada para a decomposição LU.")
        
        A = mp.matrix(matriz)
        L, U = A.LUdecomposition()
        return [[complex(val) for val in linha] for linha in L], [[complex(val) for val in linha] for linha in U]

    # Função para realizar a decomposição QR de uma matriz
    @staticmethod
    def fatoracao_QR(matriz):
        from mpmath import mp
        A = mp.matrix(matriz)
        Q, R = A.QR()
        return [[complex(val) for val in linha] for linha in Q], [[complex(val) for val in linha] for linha in R]

    # Função para realizar a decomposição de Cholesky de uma matriz simétrica positiva definida
    @staticmethod 
    def decomposicao_cholesky(matriz):
        from mpmath import mp
        if len(matriz) != len(matriz[0]):
            raise ValueError("A matriz deve ser quadrada para a decomposição de Cholesky.")
        
        A = mp.matrix(matriz)
        L = A.cholesky()
        return [[complex(val) for val in linha] for linha in L]

    # Função para resolver um sistema de equações lineares usando o método de Gauss-Seidel
    @staticmethod
    def gauss_seidel(A, b, x0, max_iter=100, tolerancia=1e-10):
        from mpmath import mp
        A = mp.matrix(A)
        b = mp.matrix(b)
        x0 = mp.matrix(x0)
        n = len(x0)
        
        x = x0
        for iteracao in range(max_iter):
            x_anterior = x.copy()
            for i in range(n):
                soma = mp.mpc(0)
                for j in range(n):
                    if i != j:
                        soma += A[i, j] * x[j]
                x[i] = (b[i] - soma) / A[i, i]
            
            erro = mp.norm(x - x_anterior, 'inf')
            if erro < tolerancia:
                print(f'Convergiu em {iteracao + 1} iterações.')
                return [complex(val) for val in x]
        
        print('Não convergiu após as iterações máximas.')
        return [complex(val) for val in x]


    @staticmethod
    def interpolar_polinomial(pontos):
        n = len(pontos)
        if n < 2:
            raise ValueError("A interpolação polinomial requer pelo menos 2 pontos.")
        
        # Separa os pontos em listas de x e y
        x, y = zip(*pontos)
        
        # Implementação da interpolação polinomial utilizando o método de Lagrange
        def lagrange_basis(i):
            def basis(x_value):
                result = 1.0
                for j in range(n):
                    if i != j:
                        result *= (x_value - x[j]) / (x[i] - x[j])
                return result
            return basis

        def polynomial_interpolation(x_value):
            interpolation = 0.0
            for i in range(n):
                interpolation += y[i] * lagrange_basis(i)(x_value)
            return interpolation
        
        return polynomial_interpolation

    # Função para ajuste de curvas usando regressão linear
    @staticmethod
    def regressao_linear(pontos):
        n = len(pontos)
        if n < 2:
            raise ValueError("O ajuste de curvas por regressão linear requer pelo menos 2 pontos.")
        # Separa os pontos em listas de x e y
        x, y = zip(*pontos)

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        sum_xy = sum(xi * yi for xi, yi in pontos)
        sum_x_squared = sum(xi ** 2 for xi in x)

        slope = (n * sum_xy - sum(x) * sum(y)) / (n * sum_x_squared - sum(x) ** 2)
        intercept = mean_y - slope * mean_x

        def linear_regression(x_value):
            return slope * x_value + intercept
        
        return linear_regression

    # Função para calcular integrais definidas usando o método do trapézio
    @staticmethod
    def integracao_trapezio(funcao, limite_inferior, limite_superior, numero_trapezios):
        if numero_trapezios < 1:
            raise ValueError("O número de trapézios deve ser pelo menos 1.")

        h = (limite_superior - limite_inferior) / numero_trapezios
        integral = (funcao(limite_inferior) + funcao(limite_superior)) / 2

        for i in range(1, numero_trapezios):
            x = limite_inferior + i * h
            integral += funcao(x)

        integral *= h
        return integral

    # Função para resolver equações diferenciais usando o método de Euler
    @staticmethod
    def metodo_euler(derivada, condicao_inicial, intervalo, passo):
        t = intervalo[0]
        y = condicao_inicial
        resultado = [(t, y)]
        
        while t + passo <= intervalo[1]:
            y += passo * derivada(t, y)
            t += passo
            resultado.append((t, y))
        
        return resultado

    @staticmethod
    def multiplicar_matriz(matriz1, matriz2):
        if len(matriz1[0]) != len(matriz2):
            raise ValueError("O número de colunas da matriz1 deve ser igual ao número de linhas da matriz2.")
        resultado = []
        for i in range(len(matriz1)):
            linha_resultado = []
            for j in range(len(matriz2[0])):
                elemento = sum(matriz1[i][k] * matriz2[k][j] for k in range(len(matriz2)))
                linha_resultado.append(elemento)
            resultado.append(linha_resultado)
        return resultado

    def produto_interno_complexo(self, v1, v2):
    # Calcula o produto interno entre dois vetores complexos
        return sum(v1_i.conjugate() * v2_i for v1_i, v2_i in zip(v1, v2))


    def vetor_conjugado_complexo(v):
        # Retorna o vetor conjugado de um vetor complexo
        return [v_i.conjugate() for v_i in v]



    def transformacao_linear(matriz, vetor):
        try:
            import numpy as np
        # Realiza a multiplicação de uma matriz por um vetor para representar a transformação linear
            return np.dot(matriz, vetor)
        except Exception as e:
            return e


    def eh_diagonalizavel(self,matriz):
        try:
            import numpy as np
            # Verifica se uma matriz é diagonalizável
            autovalores, _ = np.linalg.eig(matriz)
            return len(set(autovalores)) == matriz.shape[0]
        except Exception as e:
            return e
        
    @staticmethod
    def diagonalizar_matriz(self,matriz):
        try:
            import numpy as np
            # Diagonaliza uma matriz e retorna a matriz diagonal e a matriz de autovetores
            if self.eh_diagonalizavel(matriz):
                autovalores, autovetores = np.linalg.eig(matriz)
                matriz_diagonal = np.diag(autovalores)
                return autovetores, matriz_diagonal, np.linalg.inv(autovetores)
            else:
                raise ValueError("A matriz não é diagonalizável.")
        except Exception as e:
            return  e
        
    def sao_ortogonais(self, v1, v2):
        import math
    # Verifica se dois vetores são ortogonais
        return math.isclose(self.produto_interno_complexo(v1, v2), 0)

    def projecao_ortogonal(self, v, u):
        try:
        # Calcula a projeção ortogonal do vetor v no vetor u
            fator = self.produto_interno_complexo(v, u) / self.produto_interno_complexo(u, u)
            return [fator * u_i for u_i in u]
        except Exception as e:
            return e



    def decomposicao_valores_singulares(matriz):
        try:
            import numpy as np
            # Realiza a decomposição em valores singulares de uma matriz
            U, S, Vh = np.linalg.svd(matriz)
            return U, np.diag(S), Vh
        except Exception as e:
            return e



    def autovalores_generalizados(*args):
        try:
            import numpy as np
            # Verifica se foi fornecido pelo menos duas matrizes
            if len(args) < 2:
                raise ValueError("É necessário fornecer pelo menos duas matrizes.")
            
            # Calcula os autovalores generalizados para o par de matrizes
            eigenvalues, _ = np.linalg.eig(args[0], args[1])
            return eigenvalues
        except Exception as e:
            return str(e)  


    def decomposicao_espectral(matriz):
        try:
            import numpy as np
            # Realiza a decomposição espectral de uma matriz simétrica
            autovalores, autovetores = np.linalg.eigh(matriz)
            matriz_diagonal = np.diag(autovalores)
            return autovetores, matriz_diagonal, np.linalg.inv(autovetores)
        except Exception as e:
            return e
        
    def interpolacao_polinomial(pontos):
        try:
            import numpy as np
            # Realiza a interpolação polinomial de um conjunto de pontos
            x, y = zip(*pontos)
            coeficientes = np.polyfit(x, y, len(pontos) - 1)
            return np.poly1d(coeficientes)
        except Exception as e:
            return e

    def ajuste_mmq(x, y, grau):
        try:
            import numpy as np
            # Realiza o ajuste de curvas por mínimos quadrados a um polinômio de grau 'grau'
            coeficientes = np.polyfit(x, y, grau)
            return np.poly1d(coeficientes)
        except Exception as e:
            return e


    def decomposicao_svd(matriz):
        try:
            import numpy as np
            # Realiza a decomposição de valores singulares (SVD) de uma matriz
            U, S, Vt = np.linalg.svd(matriz)
            return U, S, Vt
        except Exception as e:
            return e
        

    def pseudo_inversa_moore_penrose(matriz):
        try:
            import numpy as np
            # Calcula a pseudo-inversa de Moore-Penrose de uma matriz
            return np.linalg.pinv(matriz)
        except Exception as e:
            return e


    def determinante_vandermonde(vetor):
        try:     
            import numpy as np
            # Calcula o determinante de uma matriz de Vandermonde construída a partir de um vetor
            return np.linalg.det(np.vander(vetor))
        except Exception as e:
            return e





    def produto_tensorial(*args):
        try:
            import numpy as np
            if len(args) < 2:
                raise ValueError("A função requer pelo menos duas matrizes como argumento.")
            
            # Inicializa a matriz resultante com a primeira matriz
            resultado = args[0]

            # Calcula o produto tensorial com as matrizes restantes
            for matriz in args[1:]:
                resultado = np.kron(resultado, matriz)

            return resultado
        except Exception as e:
            return str(e)  # Convertendo a exceção para string para retornar a mensagem de erro



    def norma_matricial(matriz, ordem=2):
        try:
            import numpy as np
            # Calcula a norma matricial de uma matriz
            return np.linalg.norm(matriz, ord=ordem)
        except Exception as e:
            return e




    def matriz_cofatora(matriz):
        try:
            import numpy as np
            # Calcula a matriz cofatora de uma matriz quadrada
            return np.linalg.inv(matriz).T * np.linalg.det(matriz)
        except Exception as e:
            return e


        

    def traco_matriz(matriz):
        try:
            import numpy as np
            # Calcula o traço de uma matriz (soma dos elementos da diagonal principal)
            return np.trace(matriz)
        except Exception as e:
            return e
        


    def decomposicao_schur(matriz):
        try:
            import numpy as np
            # Calcula a Decomposição de Schur de uma matriz
            T, Q = np.linalg.schur(matriz)
            return T, Q
        except Exception as e:
            return e



    def pseudoinversa_matriz(matriz):
        try:
            import numpy as np
            # Calcula a pseudoinversa de uma matriz
            return np.linalg.pinv(matriz)
        except Exception as e:
            return e
        

    def exp_matriz_comutador(*matrizes):
        try:
            import numpy as np
            if len(matrizes) < 2:
                raise ValueError("Pelo menos duas matrizes devem ser fornecidas.")
                
            comutador = np.zeros_like(matrizes[0])
            for i in range(len(matrizes) - 1):
                for j in range(i + 1, len(matrizes)):
                    comutador += np.dot(matrizes[i], matrizes[j]) - np.dot(matrizes[j], matrizes[i])
                    
            return np.linalg.expm(comutador)
        except Exception as e:
            return e
        


    def decomposicao_jordan(matriz):
        try:
            import numpy as np
            # Calcula a Decomposição de Jordan de uma matriz
            blocos, jordan_form = np.linalg.jordan(matriz)
            return blocos, jordan_form
        except Exception as e:
            return e
        


    def resolver_sistema_nao_linear(funcoes, valores_iniciais):
        try:
            import numpy as np
            from scipy.optimize import fsolve
            # Resolve um sistema de equações não-lineares usando o método de Newton-Raphson
            return fsolve(funcoes, valores_iniciais)
        except Exception as e:
            return e
        


    def e_conexo(grafo):
        try:
            import networkx as nx
            G = nx.Graph(grafo)
            return nx.is_connected(G)
        except Exception as e:
            return e