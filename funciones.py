
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Proyecto: Proyecto final de materia Análisis Estadístico Multivariado                               -- #
# -- Codigo: funciones.py - codigo con funciones de modelos y operaciones complejas                      -- #
# -- Repositorio: https://github.com/IFFranciscoME/FinTechLab/tree/master/MCD-ITESO/AEM/aem_proyecto     -- #
# -- Autor: Francisco ME                                                                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import numpy as np                                     # funciones numericas
import pandas as pd                                    # dataframes y utilidades
from statsmodels.tsa.api import acf, pacf              # funciones de econometria
from statsmodels.formula.api import ols                # herramientas estadisticas: modelo lineal con ols
from sklearn.preprocessing import StandardScaler       # estandarizacion de variables
from sklearn.decomposition import PCA                  # analisis de componentes principales (PCA)
import statsmodels.api as sm                           # utilidades para modelo regresion lineal
from sklearn.model_selection import train_test_split   # separacion de conjunto de entrenamiento y prueba

import mass_ts as mts

pd.set_option('display.max_rows', None)                # sin limite de renglones maximos para mostrar pandas
pd.set_option('display.max_columns', None)             # sin limite de columnas maximas para mostrar pandas
pd.set_option('display.width', None)                   # sin limite el ancho del display
pd.set_option('display.expand_frame_repr', False)      # visualizar todas las columnas de un dataframe
pd.options.mode.chained_assignment = None              # para evitar el warning enfadoso de indexacion


# -- ------------------------------------------------------------------------------- FUNCION: STS - MASS -- #
# -- ------------------------------------------------------------------------------------ Version manual -- #

def f_stsc_mass(p_precios, p_calendario, p_escenario, p_indicador, p_ventana):
    """
    :param: p_precios : dataframe : precios OHLC para crear una serie
    :param: p_calendario : dataframe :
    :param: p_escenario : dataframe :
    :param: p_indicador : dataframe :
    :param: p_ventana : int :

    :return:

    p_precios = df_pe_m1
    p_indicador = df_ce_w
    p_ventana = 40

    """

    # de la serie de precios completa de 1 minuto, tomando una sub-serie de ventana T de la reacción
    # del precio ante el comunicado de un indicador que paso la prueba de autoanova, quiero saber si
    # hubo otras veces donde se repitió el mismo patrón y si las hubo, que sean de inmediato posterior
    # a las fechas cercanas al comunicado del indicador en otras fechas.

    # fecha inicial de serie query es un evento arbitrario del calendario
    fecha_ini = p_calendario[p_calendario['escenario'] == p_escenario]['timestamp'][0]
    ind_ini = np.where(p_precios['timestamp'] == fecha_ini)[0]
    # fecha final es la fecha inicial mas un tamaño de ventana arbitrario
    ind_fin = ind_ini + p_ventana
    # se construye la serie query
    serie_q = p_precios.iloc[ind_ini[0]:ind_fin[0], :]
    serie_q = np.array((serie_q['close'] - serie_q['open']) * 10000)
    # visualizar la serie query

    # se construye la serie de busqueda (un array de numpy de 1 dimension)
    serie = np.array((p_precios['close'] - p_precios['open']) * 10000)

    # parametros del algoritmo
    # tamaño de ventana para iterar la busqueda
    batch_size = 3000
    # regresar los Top X casos que "mas se parezcan"
    top_matches = 180

    # regresar los indices y las distancias
    best_indices, best_dists = mts.mass2_batch(serie, serie_q, batch_size=batch_size, top_matches=top_matches)

    # obtener las fechas de los indices regresados
    fechas = [p_precios['timestamp'][best_indices[i]] for i in range(0, len(best_dists))]

    # para cada fecha agregar tamaño de ventana de serie query y hacer ventanas para visualizar
    # series de tiempo encontradas como similares a la original

    # iterar con todos los patrones de serie query, para cada ocurrencia del escenario elegido
    # para el indicador elegido, buscando saber si hubo repeticiones, si las hubo regresar la fechas

    return best_indices, best_dists


# -- ------------------------------------------------- FUNCION: Clasificacion de escenarios de indicador -- #
# -- ------------------------------------------------------------------------------------ Version manual -- #

def f_escenario(p0_datos):
    """
    :param: pd.DataFrame : DataFrame : calendario economico con columnas: timestamp, name,
                                                                     actual, consensus, previous
    :return:
    """

    datos = p0_datos

    # criterio para rellenar datos faltantes 'nan'
    # cuando falta en consensus
    nan_consensus = datos.index[np.isnan(datos['consensus'])].tolist()

    # asignarle a consensus lo que tiene previous
    datos['consensus'][nan_consensus] = datos['previous'][nan_consensus]

    # inicializar la columna escenario, habra los siguientes: A, B, C, D, E, F, G, H
    datos['escenario'] = ''

    # -- -- A: actual >= previous & actual >= consensus & consensus >= previous
    datos['escenario'][((datos['actual'] >= datos['previous']) &
                        (datos['actual'] >= datos['consensus']))] = 'A'

    # -- -- B: actual >= previous & actual >= consensus & consensus < Precious
    datos['escenario'][((datos['actual'] >= datos['previous']) &
                        (datos['actual'] < datos['consensus']))] = 'B'

    # -- -- C: actual >= previous & actual < consensus & consensus >= previous
    datos['escenario'][((datos['actual'] < datos['previous']) &
                        (datos['actual'] >= datos['consensus']))] = 'C'

    # -- -- D: actual >= previous & actual < consensus & consensus < previous
    datos['escenario'][((datos['actual'] < datos['previous']) &
                        (datos['actual'] < datos['consensus']))] = 'D'

    return datos


# -- ---------------------------------------- FUNCION: Generacion de variables EXOGENAS series de tiempo -- #
# -- ------------------------------------------------------------------------------------ Version manual -- #

def f_anova(p_datos_ce, p_datos_ph):
    """
    :param p_datos_ph:
    :param p_datos_ce: DataFrame : calendario economico con columnas: timestamp, name,
                                                                     actual, consensus, previous
    :return:

    p_datos_ce = df_ce_w
    p_datos_ph = df_pe_m1
    """

    def f_reaccion(p0_i, p1_ad, p2_ph, p3_ce):
        """
        :param p0_i: int : indexador para iterar en los datos de entrada
        :param p1_ad: int : cantidad de precios futuros que se considera la ventana (t + p1_ad)
        :param p2_ph: DataFrame : precios en OHLC con columnas: timestamp, open, high, low, close
        :param p3_ce: DataFrame : calendario economico con columnas: timestamp, name,
                                                                     actual, consensus, previous

        :return: resultado: diccionario con resultado final con 3 elementos como resultado

        # debugging
        p0_i = 16
        p0_ad = 30
        p2_ph = pd.DataFrame({'timestamp': 2009-01-06 05:00:00,
                              'open': 1.3556, 'high': 1.3586, 'low': 1.3516, 'close': 1.3543})
        p3_ce = pd.DataFrame({})
        """
        print(p0_i)
        # print(' fecha ce: ' + str(p3_ce['timestamp'][p0_i]))

        # Encontrar indice donde el timestamp de precios sea igual al del calendario
        indice_1 = np.where(p2_ph['timestamp'] == p3_ce['timestamp'][p0_i])[0][0]
        indice_2 = indice_1 + p1_ad
        ho = round((max(p2_ph['high'][indice_1:indice_2]) - p2_ph['open'][indice_1]) * 10000, 2)
        ol = round((p2_ph['open'][indice_1] - min(p2_ph['low'][indice_1:indice_2])) * 10000, 2)
        co = round((p2_ph['close'][indice_1] - min(p2_ph['open'][indice_1:indice_2])) * 10000, 2)

        # diccionario con resultado final
        resultado = {'ho': ho, 'ol': ol, 'co': co}

        return resultado

    # cantidad de precios a futuro a considerar
    psiguiente = 30

    # reaccion del precio para cada escenario
    d_reaccion = [f_reaccion(p0_i=i, p1_ad=psiguiente, p2_ph=p_datos_ph, p3_ce=p_datos_ce)
                  for i in range(0, len(p_datos_ce['timestamp']))]

    # print('exito')
    # acomodar resultados en columnas
    p_datos_ce['ho'] = [d_reaccion[j]['ho'] for j in range(0, len(p_datos_ce['timestamp']))]
    p_datos_ce['ol'] = [d_reaccion[j]['ol'] for j in range(0, len(p_datos_ce['timestamp']))]
    p_datos_ce['co'] = [d_reaccion[j]['co'] for j in range(0, len(p_datos_ce['timestamp']))]

    # Para cada indicador encontrar el escenario preponderante, después dividir en 4 grupos iguales los datos
    # hacer 4 grupos para prueba de anova, luego aplicar anova y regresar los indicadores en los que
    # se acepta la H0: "no hay diferencia significativa entre reacciones" del precio ante distintos escen.

    p_datos_ce = f_escenario(p0_datos=p_datos_ce)
    indicadores = list(set(p_datos_ce['Name']))
    ocurrencias = list()

    for i in range(0, len(indicadores)):
        print(i)
        datos_ind = p_datos_ce[p_datos_ce['Name'] == indicadores[i]]
        conteos = list(datos_ind['escenario'].value_counts())
        con_max = int(np.trunc(max(conteos)/4))
        arg_max = np.argmax(list(datos_ind['escenario'].value_counts()))
        esc_max = list(datos_ind['escenario'].value_counts().index)[arg_max]
        datos_ind = datos_ind[datos_ind['escenario'] == esc_max]
        datos_ind = datos_ind.tail(con_max*4)
        datos_ind['sub_escenario'] = [ele for ele in [1, 2, 3, 4] for _ in range(con_max)]

        # crear cuadro de datos para modelo ANOVA
        df_data_anova = datos_ind[['escenario', 'co', 'sub_escenario']]
        # ajustar modelo lineal para (high - open)
        modelo = ols('co ~ C(sub_escenario)', data=df_data_anova).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        res = anova_table['PR(>F)'][0]
        if res > 0.05:
            ocurrencias.append({'indicador': indicadores[i], 'escenario': esc_max,
                                'anova': anova_table})

    return pd.DataFrame({'indicador': [ocurrencias[i]['indicador'] for i in range(0, len(ocurrencias))],
                         'escenario': [ocurrencias[i]['escenario'] for i in range(0, len(ocurrencias))]})


# -- --------------------------------------- FUNCION: Generacion de variables ENDOGENAS series de tiempo -- #
# -- ------------------------------------------------------------------------------------ Version manual -- #

def f_features_end(p_datos):
    """
    :param p_datos: pd.DataFrae : dataframe con 5 columnas 'timestamp', 'open', 'high', 'low', 'close'
        :return: r_features : dataframe con 5 columnas, nombres cohercionados + Features generados

    # Debuging
    p_datos = df_precios
    p_datos = pd.DataFrame({''timestamp': {}, 'open': np.random.normal(1.1400, 0.0050, 20).
                                              'high': np.random.normal(1.1400, 0.0050, 20),
                                              'low': np.random.normal(1.1400, 0.0050, 20),
                                              'close': np.random.normal(1.1400, 0.0050, 20)})
    """

    datos = p_datos
    datos.columns = ['timestamp', 'open', 'high', 'low', 'close']

    cols = list(datos.columns)[1:]
    datos[cols] = datos[cols].apply(pd.to_numeric, errors='coerce')

    # formato columna timestamp como 'datetime'
    datos['timestamp'] = pd.to_datetime(datos['timestamp'])
    # datos['timestamp'] = datos['timestamp'].dt.tz_localize('UTC')

    # rendimiento logaritmico de ventana 1
    datos['logrend'] = np.log(datos['close']/datos['close'].shift(1)).dropna()

    # pips descontados al cierre
    datos['co'] = (datos['close']-datos['open'])*10000

    # pips descontados alcistas
    datos['ho'] = (datos['high'] - datos['open'])*10000

    # pips descontados bajistas
    datos['ol'] = (datos['open'] - datos['low'])*10000

    # pips descontados en total (medida de volatilidad)
    datos['hl'] = (datos['high'] - datos['low'])*10000

    # funciones de ACF y PACF para determinar ancho de ventana historica
    data_acf = acf(datos['logrend'].dropna(), nlags=12, fft=True)
    data_pac = pacf(datos['logrend'].dropna(), nlags=12)
    sig = round(1.96/np.sqrt(len(datos['logrend'])), 4)

    # componentes AR y MA
    maxs = list(set(list(np.where((data_pac > sig) | (data_pac < -sig))[0]) +
                    list(np.where((data_acf > sig) | (data_acf < -sig))[0])))
    # encontrar la componente maxima como indicativo de informacion historica autorelacionada
    max_n = maxs[np.argmax(maxs)]

    # condicion arbitraria: 5 resagos minimos para calcular variables moviles
    if max_n <= 2:
        max_n = 5

    # ciclo para calcular N features con logica de "Ventanas de tamaño n"
    for n in range(0, max_n):

        # resago n de ho
        datos['lag_ho_' + str(n + 1)] = np.log(datos['ho'].shift(n + 1))

        # resago n de ol
        datos['lag_ol_' + str(n + 1)] = np.log(datos['ol'].shift(n + 1))

        # promedio movil de ventana n
        datos['ma_ol_' + str(n + 2)] = datos['ol'].rolling(n + 2).mean()

        # promedio movil de ventana n
        datos['ma_ho_' + str(n + 2)] = datos['ho'].rolling(n + 2).mean()

    # asignar timestamp como index
    datos.index = pd.to_datetime(datos['timestamp'])
    # quitar columnas no necesarias para modelos de ML
    datos = datos.drop(['timestamp', 'open', 'high', 'low', 'close', 'hl', 'logrend'], axis=1)
    # borrar columnas donde exista solo NAs
    r_features = datos.dropna(axis='columns', how='all')
    # borrar renglones donde exista algun NA
    r_features = r_features.dropna(axis='rows')
    # convertir a numeros tipo float las columnas
    r_features.iloc[:, 1:] = r_features.iloc[:, 1:].astype(float)
    # estandarizacion de todas las variables independientes
    lista = r_features[list(r_features.columns[1:])]
    r_features[list(r_features.columns[1:])] = StandardScaler().fit_transform(lista)

    return r_features


# -- ---------------------------------------------------------------------- FUNCION: Ajustar RLM a datos -- #
# -- ---------------------------------------------------------------------- ---------------------------- -- #

def f_rlm(p_datos, p_y):
    """
    :param p_datos: pd.DataFrame : DataFrame con variable "y" (1era col), y n variables "x_n" (2:n)
    :param p_y : str : nombre de la columna a elegir como variable dependiente Y
    :return:
    p_datos = df_datos
    """

    datos = p_datos

    # Reacomodar los datos como arreglos
    y_multiple = np.array(datos[p_y])
    x_multiple = np.array(datos.iloc[:, 1:])

    # datos para entrenamiento y prueba
    train_x, test_x, train_y, test_y = train_test_split(x_multiple, y_multiple,
                                                        test_size=0.8, shuffle=False)

    # Agregar interceptos a X en entrenamiento y prueba
    train_x_betha = sm.add_constant(train_x)
    test_x_betha = sm.add_constant(test_x)

    # Modelo ajustado (entrenamiento)
    modelo_train = sm.OLS(train_y, train_x_betha)
    # Resultados de ajuste de modelo (entrenamiento)
    modelo_fit_train = modelo_train.fit()

    # Modelo ajustado (prueba)
    modelo_test = sm.OLS(test_y, test_x_betha)
    # Resultados de ajuste de modelo (prueba)
    modelo_fit_test = modelo_test.fit()

    # -- Con datos de ENTRENAMIENTO
    # modelo completo resultante
    r_train_modelo = modelo_fit_train
    # summary de resultados del modelo
    r_train_summary = r_train_modelo.summary()
    # DataFrame con nombre de parametros segun dataset, nombre de parametros y pvalues segun modelo
    r_df_train = pd.DataFrame({'df_params': ['intercepto'] + list(datos.columns[1:]),
                               'm_params': r_train_modelo.model.data.param_names,
                               'pv_params': r_train_modelo.pvalues})
    # valor de AIC del modelo
    r_train_aic = r_train_modelo.aic
    # valor de BIC del modelo
    r_train_bic = r_train_modelo.bic

    # -- Con datos de PRUEBA
    # modelo completo resultante
    r_test_modelo = modelo_fit_test
    # summary de resultados del modelo
    r_test_summary = r_test_modelo.summary()
    # DataFrame con nombre de parametros segun dataset, nombre de parametros y pvalues segun modelo
    r_df_test = pd.DataFrame({'df_params': ['intercepto'] + list(datos.columns[1:]),
                              'm_params': r_test_modelo.model.data.param_names,
                              'pv_params': r_test_modelo.pvalues})
    # valor de AIC del modelo
    r_test_aic = r_test_modelo.aic
    # valor de BIC del modelo
    r_test_bic = r_test_modelo.bic

    # tabla de resultados periodo de entrenamiento
    r_df_pred_train = pd.DataFrame({'y': train_y, 'y_ajustada': modelo_fit_train.predict()})
    # tabla de resultados periodo de prueba
    r_df_pred_test = pd.DataFrame({'y': test_y, 'y_ajustada': modelo_fit_test.predict()})

    r_d_modelo = {'train': {'modelo': r_train_modelo, 'summary': r_train_summary, 'parametros': r_df_train,
                            'resultado': r_df_pred_train, 'aic': r_train_aic, 'bic': r_train_bic},
                  'test': {'modelo': r_test_modelo, 'summary': r_test_summary, 'parametros': r_df_test,
                            'resultado': r_df_pred_test, 'aic': r_test_aic, 'bic': r_test_bic}}

    return r_d_modelo


# -- ---------------------------------------------------------------------- FUNCION: Aplicar PCA a datos -- #
# -- ---------------------------------------------------------------------- ---------------------------- -- #

def f_pca(p_datos, p_exp):
    """
    :param p_datos:
    :param p_exp:
    :return:

    p_datos = df_datos
    p_exp = .90
    """
    datos = p_datos

    pca = PCA(n_components=10)
    datos_pca = datos.iloc[:, 1:]
    pca.fit(datos_pca)
    # Calcular los vectores y valores propios de la martiz de covarianza
    w, v = np.linalg.eig(pca.get_covariance())
    # ordenar los valores de mayor a menor
    indx = np.argsort(w)[::-1]
    # calcular el procentaje de varianza en cada componente
    porcentaje = w[indx] / np.sum(w)
    # calcular el porcentaje acumulado de los componentes
    porcent_acum = np.cumsum(porcentaje)
    # encontrar las componentes necesarias para lograr explicar el 90% de variabilidad
    pca_90 = np.where(porcent_acum > p_exp)[0][0] + 1

    pca = PCA(n_components=pca_90)
    datos_pca = datos.iloc[:, 1:]
    df1 = datos.iloc[:, 0]
    pca.fit(datos_pca)
    df2 = pd.DataFrame(pca.transform(datos_pca))

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    r_datos_pca = pd.concat([df1, df2], axis=1)
    r_datos_pca.index = datos_pca.index

    # Renombrar columnas
    r_datos_pca.columns = ['pca_y'] + ['pca_x_' + str(i) for i in range(0, pca_90)]

    return r_datos_pca


# -- ---------------------------------------------------------------------- FUNCION: Desempeño de modelo -- #
# -- ---------------------------------------------------------------------- ---------------------------- -- #

def f_analisis_mod(p_datos):
    """
    :param p_datos
    :return:
    p_datos = df_datos
    """

    # from statsmodels.stats.outliers_influence import variance_inflation_factor
    # from patsy import dmatrices
    # p_datos.index = [np.arange(len(p_datos))]
    # features = " + ".join(list(p_datos.columns[1:]))
    # y, X = dmatrices('co ~' + features, p_datos, return_type='dataframe')
    # vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # vif["features"] = X.columns

    return p_datos


# -- ------------------------------------------------------------------- FUNCION: Seleccion de variables -- #
# -- -------------------------------------------------------------------- ------------------------------ -- #

def f_feature_importance(p_datos):
    """
    :param p_datos:
    :return:
    p_datos = df_datos
    """

    # np.corrcoef()

    return p_datos
