# Probabilidade de um evento simples
@staticmethod
def probabilidade_evento_simples(evento_favoravel, espaco_amostral):
    return evento_favoravel / espaco_amostral

# Probabilidade complementar
@staticmethod
def probabilidade_complementar(probabilidade_evento):
    return 1 - probabilidade_evento

# Probabilidade conjunta (Eventos independentes)
@staticmethod
def probabilidade_conjunta_independentes(probabilidade_evento1, probabilidade_evento2):
    return probabilidade_evento1 * probabilidade_evento2

# Probabilidade conjunta (Eventos dependentes)
@staticmethod
def probabilidade_conjunta_dependentes(probabilidade_evento1, probabilidade_evento2_dado_evento1):
    return probabilidade_evento1 * probabilidade_evento2_dado_evento1

# Probabilidade da união de eventos mutuamente exclusivos
@staticmethod
def probabilidade_uniao_mutuamente_exclusivos(probabilidade_evento1, probabilidade_evento2):
    return probabilidade_evento1 + probabilidade_evento2

# Probabilidade da união de eventos não mutuamente exclusivos
@staticmethod
def probabilidade_uniao_nao_mutuamente_exclusivos(probabilidade_evento1, probabilidade_evento2, probabilidade_intersecao):
    return probabilidade_evento1 + probabilidade_evento2 - probabilidade_intersecao

# Probabilidade condicional
@staticmethod
def probabilidade_condicional(probabilidade_evento1, probabilidade_evento2_dado_evento1):
    return probabilidade_evento2_dado_evento1 / probabilidade_evento1

# Teorema de Bayes
@staticmethod
def teorema_bayes(probabilidade_evento1, probabilidade_evento2_dado_evento1, probabilidade_evento2):
    return (probabilidade_evento2_dado_evento1 * probabilidade_evento1) / probabilidade_evento2

# Probabilidade de interseção de eventos independentes
@staticmethod
def probabilidade_intersecao_independentes(probabilidade_evento1, probabilidade_evento2):
    return probabilidade_evento1 * probabilidade_evento2

# Probabilidade de interseção de eventos dependentes
@staticmethod
def probabilidade_intersecao_dependentes(probabilidade_evento1, probabilidade_evento2_dado_evento1):
    return probabilidade_evento1 * probabilidade_evento2_dado_evento1

# Regra da multiplicação (Eventos independentes)
@staticmethod
def regra_multiplicacao_independentes(probabilidade_eventos):
    from sympy import S, factorial, sqrt, prod, partition, integrate
    return prod(probabilidade_eventos)

# Regra da multiplicação (Eventos dependentes)
@staticmethod
def regra_multiplicacao_dependentes(probabilidade_eventos_dado_evento_anterior):
    from sympy import S, factorial, sqrt, prod, partition, integrate
    return prod(probabilidade_eventos_dado_evento_anterior)

# Probabilidade condicional múltipla
@staticmethod
def probabilidade_condicional_multipla(probabilidade_eventos):
    return probabilidade_eventos[-1] / probabilidade_eventos[:-1]

# Probabilidade de um evento composto
@staticmethod
def probabilidade_evento_composto(probabilidades_eventos):
    return sum(probabilidades_eventos)

# Probabilidade conjunta de eventos compostos independentes
@staticmethod
def probabilidade_conjunta_compostos_independentes(probabilidades_eventos):
    from sympy import S, factorial, sqrt, prod, partition, integrate
    return prod(probabilidades_eventos)

# Probabilidade conjunta de eventos compostos dependentes
@staticmethod
def probabilidade_conjunta_compostos_dependentes(probabilidades_eventos_dado_evento_anterior):
    from sympy import S, factorial, sqrt, prod, partition, integrate
    return prod(probabilidades_eventos_dado_evento_anterior)

# Probabilidade de eventos independentes não mutuamente exclusivos
@staticmethod
def probabilidade_independentes_nao_mutuamente_exclusivos(probabilidades_eventos):
    union_prob = sum(probabilidades_eventos) - sum(probabilidades_eventos[i] * probabilidades_eventos[j] for i in range(len(probabilidades_eventos)) for j in range(i + 1, len(probabilidades_eventos)))
    return union_prob
# Probabilidade de eventos dependentes não mutuamente exclusivos
@staticmethod
def probabilidade_dependentes_nao_mutuamente_exclusivos(probabilidades_eventos_dado_evento_anterior):
    intersection_prob = 1.0
    for prob in probabilidades_eventos_dado_evento_anterior:
        intersection_prob *= prob
    
    union_prob = sum(probabilidades_eventos_dado_evento_anterior) - intersection_prob
    return union_prob

# Probabilidade de uma variável aleatória contínua (densidade de probabilidade)
@staticmethod
def densidade_probabilidade_variavel_aleatoria_contínua(x, f_x):
    return f_x(x)

# Probabilidade de uma variável aleatória discreta
@staticmethod
def probabilidade_variavel_aleatoria_discreta(X, P_X):
    return sum([P_X[i] for i in X])

# Esperança matemática de uma variável aleatória discreta
@staticmethod
def esperanca_variavel_aleatoria_discreta(X, P_X):
    return sum([x * P_X[i] for i, x in enumerate(X)])


# Esperança matemática de uma variável aleatória contínua
@staticmethod
def esperanca_variavel_aleatoria_continua(f_x, a, b):
    from sympy import integrate, symbols
    x = symbols('x')
    return integrate(x * f_x, (x, a, b))
