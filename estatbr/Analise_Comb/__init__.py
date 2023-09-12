# Arranjo simples (Permutação)
def arranjo(n:int, k:int):
    import math
    return math.factorial(n) // math.factorial(n - k)

# Combinação (Binômio)
def combinacao(n:int, k:int):
    import math
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# Combinação com repetição
def combinacao_repeticao(n:int, k:int):
    import math
    def combinacao(n, k):
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
    return combinacao(n + k - 1, k)

# Permutação com repetição
def permutacao_repeticao(lista):
    import math
    divisor = 1
    for elem in set(lista):
        divisor *= math.factorial(lista.count(elem))
    return math.factorial(len(lista)) // divisor

# Permutação circular
def permutacao_circular(n:int):
    import math
    return math.factorial(n - 1)

# Números de Stirling de segunda espécie
def numeros_stirling_segunda(n, k):
    import numpy as np
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0
    else:
        s = np.zeros((n + 1, k + 1))
        s[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                s[i][j] = j * s[i - 1][j] + s[i - 1][j - 1]
        return s[n][k]

# Números de Stirling de primeira espécie
def numeros_stirling_primeira(n, k):
    import numpy as np
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0
    else:
        s = np.zeros((n + 1, k + 1))
        s[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, k + 1):
                s[i][j] = (i - 1) * s[i - 1][j] + s[i - 1][j - 1]
        return s[n][k]

# Coeficiente multinomial
def multinomial(*args):
    import math
    numerator = math.factorial(sum(args))
    denominator = 1
    for arg in args:
        denominator *= math.factorial(arg)
    return numerator // denominator

# Número de Bell
def numero_bell(n):
    bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    bell[0][0] = 1

    for i in range(1, n + 1):
        bell[i][0] = bell[i - 1][i - 1]
        for j in range(1, i + 1):
            bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]

    return bell[n][0]

# Coeficiente binomial generalizado
def coeficiente_binomial_generalizado(n, k, m):
    import math
    return math.factorial(n) // (math.factorial(k) * math.factorial(m) * math.factorial(n - k - m))

# Número de derangements
def numero_derangement_biblioteca(n):
    import math
    return math.factorial(n) * sum((-1)**i / math.factorial(i) for i in range(n + 1))


# Triângulo de Pascal
def triangulo_pascal(n:int):
    pascal = [[1]]
    for i in range(1, n + 1):
        linha = [1]
        for j in range(1, i):
            linha.append(pascal[i - 1][j - 1] + pascal[i - 1][j])
        linha.append(1)
        pascal.append(linha)
    return pascal

# Números harmônicos
def numeros_harmonicos(n):
    return sum([1/i for i in range(1, n + 1)])

# Número de Fibonacci
def fibonacci(n):
    from sympy import fibonacci
    return fibonacci(n)

# Número de Lucas
def lucas(n):
    from sympy import lucas
    return lucas(n)

# Números de Catalão
def numeros_catalao(n):
    from sympy import catalan
    return catalan(n)

# Sequência de Farey
def sequencia_farey(n):
    farey = set()
    for denominator in range(1, n + 1):
        for numerator in range(denominator + 1):
            farey.add(numerator / denominator)
    return sorted(list(farey))

# Número de Partições


def numero_particoes(n):
    from sympy import partition
    return partition(n)


# Número de Stirling de segunda espécie (versão iterativa)
def numeros_stirling_segunda_iterativo(n, k):
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0

    stirling = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        stirling[i][0] = 0
    for i in range(k + 1):
        stirling[0][i] = 0

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            if i == j:
                stirling[i][j] = 1
            else:
                stirling[i][j] = j * stirling[i - 1][j] + stirling[i - 1][j - 1]

    return stirling[n][k]

# Número de Stirling de primeira espécie (versão iterativa)
def numeros_stirling_primeira_iterativo(n, k):
    if n == k == 0:
        return 1
    elif n == 0 or k == 0:
        return 0

    stirling = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        stirling[i][0] = 0
    for i in range(k + 1):
        stirling[0][i] = 0

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            if i == j:
                stirling[i][j] = 1
            else:
                stirling[i][j] = (i - 1) * stirling[i - 1][j] + stirling[i - 1][j - 1]

    return stirling[n][k]

# Coeficiente multinomial (versão iterativa)
def multinomial_iterativo(*args):
    import math
    n = sum(args)
    numerator = 1
    for i in range(n, n - sum(args), -1):
        numerator *= i
    denominator = math.prod([math.factorial(arg) for arg in args])
    return numerator // denominator

# Número de Bell (versão iterativa)
def numero_bell_iterativo(n):
    bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    bell[0][0] = 1

    for i in range(1, n + 1):
        bell[i][0] = bell[i - 1][i - 1]
        for j in range(1, i + 1):
            bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]

    return bell[n][0]



# Número de derivações
def numero_derivacoes(n):
    import scipy.special as sp
    return sum([(-1)**k * sp.comb(n, k) * (n - k)**n for k in range(n + 1)])

# Coeficiente binomial generalizado (versão iterativa)
def coeficiente_binomial_generalizado_iterativo(n, k, m):
    import math
    numerator = 1
    for i in range(n, n - k, -1):
        numerator *= i
    denominator = math.prod([math.factorial(arg) for arg in [k, m, n - k - m]])
    return numerator // denominator

# Número de derangements (versão iterativa)
def numero_derangement_iterativo(n):
    if n == 0:
        return 1
    elif n == 1:
        return 0

    derangement = [0] * (n + 1)
    derangement[0], derangement[1] = 1, 0

    for i in range(2, n + 1):
        derangement[i] = (i - 1) * (derangement[i - 1] + derangement[i - 2])

    return derangement[n]

# Números harmônicos (versão iterativa)
def numeros_harmonicos_iterativo(n):
    return sum([1/i for i in range(1, n + 1)])

# Número de Fibonacci (versão iterativa)
def fibonacci_iterativo(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    fib = [0] * (n + 1)
    fib[1] = 1

    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n]

# Número de Lucas (versão iterativa)
def lucas_iterativo(n):
    if n == 0:
        return 2
    elif n == 1:
        return 1

    lucas = [0] * (n + 1)
    lucas[1] = 1
    lucas[2] = 2

    for i in range(3, n + 1):
        lucas[i] = lucas[i - 1] + lucas[i - 2]

    return lucas[n]

# Números de Catalão (versão iterativa)
def numeros_catalao_iterativo(n):
    catalan = [0] * (n + 1)
    catalan[0] = 1

    for i in range(1, n + 1):
        for j in range(i):
            catalan[i] += catalan[j] * catalan[i - j - 1]

    return catalan[n]

# Sequência de Farey (versão iterativa)
def sequencia_farey_iterativo(n):
    farey = set()
    for denominator in range(1, n + 1):
        for numerator in range(1, denominator + 1):
            farey.add(numerator / denominator)
    return sorted(list(farey))

# Número de Partições (versão iterativa)
def numero_particoes_iterativo(n, m):
    partitions = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        partitions[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if j >= i:
                partitions[i][j] = partitions[i - 1][j] + partitions[i][j - i]
            else:
                partitions[i][j] = partitions[i - 1][j]

    return partitions[n][m]

