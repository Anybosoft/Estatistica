
## estatbr Library

The "estatbr" library is a Python library for basic/advance statistical calculations. It provides a collection of functions for various statistical operations.

### Functions:

1. **media(args: Sequence[int | float]) -> float**

   Calculates the mean (average) of a sequence of numbers.

2. **mediana(args: Sequence[int | float]) -> float**

   Computes the median of a sequence of numbers.

3. **moda(args: Sequence[int | float])**

   Calculates the mode(s) of a sequence of numbers.

4. **desvio_padrao(args: Sequence[int | float]) -> float**

   Computes the standard deviation of a sequence of numbers.

5. **desvio_medio(args: Sequence[int | float]) -> float**

   Calculates the mean absolute deviation of a sequence of numbers.

6. **variancia(args: Sequence[int | float]) -> float**

   Computes the variance of a sequence of numbers.

7. **comparar(a: List[int | float], b: List[int | float]) -> float**

   Compares two lists of numbers and returns a measure of similarity.

8. **media_ponderada(valores: List[int | float], pesos: List[int | float]) -> float**

   Computes the weighted mean of a list of values using given weights.

9. **media_geometrica(args: Sequence[int | float]) -> float**

   Calculates the geometric mean of a sequence of numbers.

10. **media_quadratica(args: Sequence[int | float]) -> float**

    Computes the root mean square (quadratic mean) of a sequence of numbers.

11. **intervalo_medio(args: Sequence[int | float]) -> float**

    Computes the midrange (average of the minimum and maximum) of a sequence of numbers.

12. **intervalo_medio_entre_dois_numeros(a, b) -> float**

    Calculates the midpoint between two numbers, 'a' and 'b'.

13. **amplitude(args: Sequence[int | float])**

    Calculates the range (difference between maximum and minimum) of a sequence of numbers.

14. **quartis(args: Sequence[int | float])**

    Computes the quartiles (Q1, Q2, Q3) of a sequence of numbers.

15. **amplitude_interquartil(args: Sequence[int | float])**

    Calculates the interquartile range (IQR) of a sequence of numbers.

16. **coeficiente_correlacao(x, y)**

    Calculates the Pearson correlation coefficient between two lists of values 'x' and 'y'.

17. **regressao_linear(x: List[int | float], y: List[int | float])**

    Performs linear regression on two lists of values 'x' and 'y' and returns the coefficients of the regression line.

18. **coeficiente_variacao(args: Sequence[int | float])**

    Computes the coefficient of variation (CV) for a sequence of numbers.

19. **media_harmonica(args: Sequence[int | float])**

    Calculates the harmonic mean of a sequence of numbers.

20. **distribuicao_frequencia(dados: List[int | float], num_classes)**

    Constructs a frequency distribution table for a list of data.

21. **intervalo_confianca(dados: List[int | float], nivel_confianca)**

    Computes the confidence interval for a list of data with a specified confidence level.

22. **coeficiente_assimetria(args: Sequence[int | float])**

    Calculates the skewness (measure of asymmetry) of a sequence of numbers.

23. **curtose(args: Sequence[int | float])**

    Computes the kurtosis (measure of the "tailedness") of a sequence of numbers.

24. **coeficiente_correlacao_pearson(x: List[int | float], y: List[int | float])**

    Calculates the Pearson correlation coefficient between two lists of values 'x' and 'y'.

25. **teste_t(amostra1: Sequence[int | float], amostra2: Sequence[int | float])**

    Performs a t-test to compare two samples and returns the t-statistic.

26. **teste_qui_quadrado(freq_obs, freq_esp)**

    Performs a chi-squared test of independence between observed and expected frequencies in contingency tables.

27. **analise_variancia(args: Sequence[int | float])**

    Performs an analysis of variance (ANOVA) on multiple groups of data and returns the F-statistic.

28. **teste_normalidade(amostra: List[int | float], alpha=0.05) -> float**

    Performs a chi-squared goodness-of-fit test for normality on a sample of data.

29. **teste_homogeneidade(*grupos: Sequence[int | float], alpha=0.05)**

    Performs a one-way analysis of variance (ANOVA) to test the equality of means across multiple groups.

These functions cover a wide range of statistical calculations and can be useful for various data analysis tasks. Please refer to the function descriptions for more details on their usage.

Please note that this library may raise exceptions if input data does not meet certain requirements, such as empty lists or different sample sizes for certain functions. Make sure to handle exceptions appropriately when using these functions in your code.