def grafico_barra(x=None, y=None, eixox=None, eixoy=None, titulo=None, cor=None, legenda=None, tamanho_d_figura=(8, 6), salvar_como=None, xlimite=None, ylimite=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        if x is not None and y is not None:
            plt.figure(figsize=tamanho_d_figura)
            bars = plt.bar(x, y, color=cor)

            if eixox:
                plt.xlabel(eixox)
            if eixoy:
                plt.ylabel(eixoy)
            if titulo:
                plt.title(titulo)
            if legenda:
                plt.legend(bars, legenda)

            if xlimite:
                plt.xlim(xlimite)
            if ylimite:
                plt.ylim(ylimite)

            if salvar_como:
                plt.savefig(salvar_como)
            else:
                plt.show()

        else:
            # Aqui você pode adicionar alguma lógica para lidar com o caso em que x e y não são fornecidos.
            # Por exemplo, levantar um erro ou exibir uma mensagem informativa.
            pass

    except Exception as e:
        return e

def grafico_dispersao(x=None, y=None, eixox=None, eixoy=None, titulo=None, cor=None, marcador='o', tamanho_d_figura=(8, 6), salvar_como=None, xlimite=None, ylimite=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        if x is not None and y is not None:
            plt.figure(figsize=tamanho_d_figura)
            plt.scatter(x, y, color=cor, marker=marcador)

            if eixox:
                plt.xlabel(eixox)
            if eixoy:
                plt.ylabel(eixoy)
            if titulo:
                plt.title(titulo)

            if xlimite:
                plt.xlim(xlimite)
            if ylimite:
                plt.ylim(ylimite)

            if salvar_como:
                plt.savefig(salvar_como)
            else:
                plt.show()

        else:
            # Aqui você pode adicionar alguma lógica para lidar com o caso em que x e y não são fornecidos.
            # Por exemplo, levantar um erro ou exibir uma mensagem informativa.
            pass

    except Exception as e:
        return e

def grafico_pizza(dados=None, tamanhos=None, titulo=None, cor=None, destaca=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        if dados is not None and tamanhos is not None:
            plt.figure(figsize=tamanho_d_figura)
            plt.pie(tamanhos, labels=dados, colors=cor, explode=destaca, autopct='%1.1f%%', shadow=True)

            if titulo:
                plt.title(titulo)

            if salvar_como:
                plt.savefig(salvar_como)
            else:
                plt.show()

        else:
            # Aqui você pode adicionar alguma lógica para lidar com o caso em que labels e sizes não são fornecidos.
            # Por exemplo, levantar um erro ou exibir uma mensagem informativa.
            pass

    except Exception as e:
        return e

def grafico_linhas(x, y, titulo=None, eixox=None, eixoy=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.plot(x, y)

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if eixoy:
            plt.ylabel(eixoy)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e

def grafico_area(x, y, titulo=None, eixox=None, eixoy=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.fill_between(x, y)

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if eixoy:
            plt.ylabel(eixoy)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e
    


def grafico_histograma(valores, titulo=None, eixox=None, eixoy=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.hist(valores, bins='auto', alpha=0.7, color='blue', edgecolor='black')

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if eixoy:
            plt.ylabel(eixoy)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e



def grafico_boxplot_(dados, titulo=None, eixox=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.boxplot(dados)

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e



def grafico_boxplot(dados, titulo=None, eixox=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.boxplot(dados)

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e

    
    

def surface_plot(x, y, z, titulo=None, eixox=None, eixoy=None, eixoz=None, tamanho_d_figura=(10, 8), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        import numpy as np
        fig = plt.figure(figsize=tamanho_d_figura)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis')

        if titulo:
            plt.title(titulo)

        if eixox:
            ax.set_xlabel(eixox)

        if eixoy:
            ax.set_ylabel(eixoy)

        if eixoz:
            ax.set_zlabel(eixoz)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e


def grafico_contorno(x, y, z, titulo=None, eixox=None, eixoy=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['toolbar'] = 'None'
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=tamanho_d_figura)
        plt.contour(X, Y, z)

        if titulo:
            plt.title(titulo)

        if eixox:
            plt.xlabel(eixox)

        if eixoy:
            plt.ylabel(eixoy)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e



def grafico_polar(angulos, valores, titulo=None, tamanho_d_figura=(8, 6), salvar_como=None):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['toolbar'] = 'None'
        plt.figure(figsize=tamanho_d_figura)
        plt.polar(angulos, valores)

        if titulo:
            plt.title(titulo)

        if salvar_como:
            plt.savefig(salvar_como)
        else:
            plt.show()

    except Exception as e:
        return e

