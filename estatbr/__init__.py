from typing import Sequence, List

def media(args: Sequence[int | float]) -> float:
    try:
        return sum(args) / len(args)
    except Exception as e:
        return e
    


def mediana(args: Sequence[int | float]) -> float:
    try:
        sorted_args = sorted(args)
        n = len(sorted_args)
        if n % 2 == 0:
            return (sorted_args[n // 2 - 1] + sorted_args[n // 2]) / 2
        else:
            return sorted_args[n // 2]
    except Exception as e:
        return e

def moda(args: Sequence[int | float]):
    try:
        counter = {}
        for num in args:
            if num in counter:
                counter[num] += 1
            else:
                counter[num] = 1
        max_count = max(counter.values())
        return [num for num, count in counter.items() if count == max_count]
    except Exception as e:
        return e

def desvio_padrao(args: Sequence[int | float]) -> float:
    try:
        import math
        n = len(args)
        if n == 0:
            raise ValueError("A lista de valores não pode ser vazia.")
        mean = sum(args) / n
        squared_diff_sum = sum((x - mean) ** 2 for x in args)
        variance = squared_diff_sum / n
        return math.sqrt(variance)
    except Exception as e:
        return e

def desvio_medio(args: Sequence[int | float]) -> float:
    try:
        n = len(args)
        if n == 0:
            raise ValueError("A lista de valores não pode ser vazia.")
        return sum(abs(x - sum(args) / n) for x in args) / n
    except Exception as e:
        return e

def variancia(args: Sequence[int | float]) -> float:
    try:
        n = len(args)
        if n == 0:
            raise ValueError("A lista de valores não pode ser vazia.")
        mean = sum(args) / n
        squared_diff_sum = sum((x - mean) ** 2 for x in args)
        return squared_diff_sum / n
    except Exception as e:
        return e

def comparar(a: List[int | float], b: List[int | float]) -> float:
    try:
        if len(a) != len(b):
            raise ValueError("As listas de valores devem ter o mesmo tamanho.")
        n = len(a)
        diff_sum = sum(abs(a[i] - b[i]) for i in range(n))
        return diff_sum / n
    except Exception as e:
        return e

def media_ponderada(valores: List[int | float], pesos: List[int | float]) -> float:
    try:
        if len(valores) != len(pesos):
            raise ValueError("As listas de valores e pesos devem ter o mesmo tamanho.")
        weighted_sum = sum(valores[i] * pesos[i] for i in range(len(valores)))
        sum_of_weights = sum(pesos)
        return weighted_sum / sum_of_weights
    except Exception as e:
        return e

def media_geometrica(args: Sequence[int | float]) -> float:
    try:
        product = 1
        for num in args:
            product *= num
        return product ** (1 / len(args))
    except Exception as e:
        return e

def media_quadratica(args: Sequence[int | float]) -> float:
    try:
        import math
        squared_sum = sum(x ** 2 for x in args)
        return math.sqrt(squared_sum / len(args))
    except Exception as e:
        return e

def intervalo_medio(args: Sequence[int | float]) -> float:
    try:
        sorted_args = sorted(args)
        return (sorted_args[0] + sorted_args[-1]) / 2
    except Exception as e:
        return e

def intervalo_medio_entre_dois_numeros(a, b) -> float:
    try:
        return (a + b) / 2
    except Exception as e:
        return e

def amplitude(args: Sequence[int | float]):
    try:
        return max(args) - min(args)
    except Exception as e:
        return e

def quartis(args: Sequence[int | float]):
    try:
        import statistics
        q1 = statistics.percentile(args, 25)
        q2 = statistics.median(args)
        q3 = statistics.percentile(args, 75)
        return q1, q2, q3
    except Exception as e:
        return e

def amplitude_interquartil(self, args: Sequence[int | float]):
    try:
        q1, _, q3 = self.quartis(args)
        return q3 - q1
    except Exception as e:
        return e



def coeficiente_correlacao(x, y):
    try:
        from scipy import stats
        import math
        if len(x) != len(y):
            raise ValueError("As listas de valores devem ter o mesmo tamanho.")
        n = len(x)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_squared = sum(x[i] ** 2 for i in range(n))
        sum_y_squared = sum(y[i] ** 2 for i in range(n))
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2))
        return numerator / denominator
    except Exception as e:
        return e

def regressao_linear(x: List[int | float], y: List[int | float]):
    try:
        if len(x) != len(y):
            raise ValueError("As listas de valores devem ter o mesmo tamanho.")
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_squared = sum(x[i] ** 2 for i in range(n))
        sum_xy = sum(x[i] * y[i] for i in range(n))

        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        b = (sum_y - a * sum_x) / n

        return a, b
    except Exception as e:
        return e

def coeficiente_variacao(args: Sequence[int | float]):
    try:
        import statistics
        mean = statistics.mean(args)
        std_deviation = statistics.stdev(args)
        return (std_deviation / mean) * 100
    except Exception as e:
        return e

def media_harmonica(args: Sequence[int | float]):
    try:
        reciprocal_sum = sum(1 / num for num in args)
        return len(args) / reciprocal_sum
    except Exception as e:
        return e
    
def distribuicao_frequencia(dados: List[int | float], num_classes):
    try:
        sorted_data = sorted(dados)
        min_value = sorted_data[0]
        max_value = sorted_data[-1]
        range_value = max_value - min_value
        class_width = range_value / num_classes

        frequency_table = {}
        for i in range(num_classes):
            lower_bound = min_value + i * class_width
            upper_bound = lower_bound + class_width
            frequency_table[(lower_bound, upper_bound)] = 0

        for value in sorted_data:
            for interval in frequency_table.keys():
                lower_bound, upper_bound = interval
                if lower_bound <= value < upper_bound:
                    frequency_table[interval] += 1
                    break

        return frequency_table
    except Exception as e:
        return e

def intervalo_confianca(dados: List[int | float], nivel_confianca):
    try:
        import statistics
        from scipy import stats
        import math
        n = len(dados)
        mean = statistics.mean(dados)
        std_deviation = statistics.stdev(dados)
        t_value = stats.t.ppf((1 + nivel_confianca) / 2, n - 1)
        margin_of_error = t_value * (std_deviation / math.sqrt(n))
        lower_bound = mean - margin_of_error
        upper_bound = mean + margin_of_error
        return lower_bound, upper_bound
    except Exception as e:
        return e

def coeficiente_assimetria(args: Sequence[int | float]):
    try:
        import math
        n = len(args)
        mean = sum(args) / n
        variance = sum((x - mean) ** 2 for x in args) / n
        std_deviation = math.sqrt(variance)
        cubed_deviations = [(num - mean) ** 3 for num in args]
        sum_cubed_deviations = sum(cubed_deviations)
        skewness = (sum_cubed_deviations / (n * std_deviation ** 3))
        return skewness
    except Exception as e:
        return e


def curtose(args: Sequence[int | float]):
    try:
        n = len(args)
        mean = sum(args) / n
        variance = sum((x - mean) ** 2 for x in args) / n
        fourth_power_deviations = [(num - mean) ** 4 for num in args]
        sum_fourth_power_deviations = sum(fourth_power_deviations)
        kurtosis = (sum_fourth_power_deviations / (n * variance ** 2)) - 3
        return kurtosis
    except Exception as e:
        return e


def coeficiente_correlacao_pearson(x: List[int | float], y: List[int | float]):
    try:
        import math
        if len(x) != len(y):
            raise ValueError("As listas de valores devem ter o mesmo tamanho.")
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x_squared = sum(x[i] ** 2 for i in range(n))
        sum_y_squared = sum(y[i] ** 2 for i in range(n))
        sum_xy = sum(x[i] * y[i] for i in range(n))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x_squared - sum_x ** 2) * (n * sum_y_squared - sum_y ** 2))
        pearson_correlation = numerator / denominator
        return pearson_correlation
    except Exception as e:
        return e



def teste_t(amostra1: Sequence[int | float], amostra2: Sequence[int | float]):
    try:
        import math
        if len(amostra1) != len(amostra2):
            raise ValueError("As amostras devem ter o mesmo tamanho.")

        n1 = len(amostra1)
        n2 = len(amostra2)

        mean1 = sum(amostra1) / n1
        mean2 = sum(amostra2) / n2

        variance1 = sum((x - mean1) ** 2 for x in amostra1) / (n1 - 1)
        variance2 = sum((x - mean2) ** 2 for x in amostra2) / (n2 - 1)

        pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
        pooled_std_deviation = math.sqrt(pooled_variance)

        t_value = (mean1 - mean2) / (pooled_std_deviation * math.sqrt(1 / n1 + 1 / n2))
        return t_value
    except Exception as e:
        return e


def teste_qui_quadrado(freq_obs, freq_esp):
    try:
        if len(freq_obs) != len(freq_esp):
            raise ValueError("As tabelas de frequência devem ter o mesmo tamanho.")
        n = len(freq_obs)
        chi_squared = sum((freq_obs[i] - freq_esp[i]) ** 2 / freq_esp[i] for i in range(n))
        return chi_squared
    except Exception as e:
        return e

def analise_variancia(args: Sequence[int | float]):
    try:
        num_amostras = len(args)
        sizes = [len(amostra) for amostra in args]
        grand_mean = sum(sum(amostra) for amostra in args) / sum(sizes)
        total_ss = sum(sum((x - grand_mean) ** 2 for x in amostra) for amostra in args)
        df_total = sum(size - 1 for size in sizes)
        df_between = num_amostras - 1
        df_within = df_total - df_between
        ss_between = sum(size * (sum(amostra) / size - grand_mean) ** 2 for size, amostra in zip(sizes, args))
        ms_between = ss_between / df_between
        ss_within = total_ss - ss_between
        ms_within = ss_within / df_within
        f_value = ms_between / ms_within
        return f_value
    except Exception as e:
        return e
    


def teste_normalidade(amostra: List[int | float], alpha=0.05) -> float:
    import numpy as np
    from scipy.stats import chi2
    n = len(amostra)
    mean = np.mean(amostra)
    std_deviation = np.std(amostra, ddof=1)
    z_score = (amostra - mean) / std_deviation
    squared_z_score = z_score ** 2
    chi_square = np.sum(squared_z_score)
    critical_value = chi2.ppf(1 - alpha, df=n - 1)

    return chi_square <= critical_value

def teste_homogeneidade(*grupos: Sequence[int | float], alpha=0.05):
    from scipy.stats import f
    import numpy as np
    n_grupos = len(grupos)
    n_total = np.sum([len(grupo) for grupo in grupos])
    mean_total = np.mean(np.concatenate(grupos))
    squared_deviations_total = np.sum([(x - mean_total) ** 2 for grupo in grupos for x in grupo])
    squared_deviations_between = np.sum([len(grupo) * (np.mean(grupo) - mean_total) ** 2 for grupo in grupos])

    df_between = n_grupos - 1
    df_within = n_total - n_grupos

    mean_squared_deviations_between = squared_deviations_between / df_between
    mean_squared_deviations_within = squared_deviations_total / df_within

    f_statistic = mean_squared_deviations_between / mean_squared_deviations_within
    critical_value = f.ppf(1 - alpha, dfn=df_between, dfd=df_within)

    return f_statistic <= critical_value


