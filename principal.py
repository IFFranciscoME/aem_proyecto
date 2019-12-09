
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Proyecto: Proyecto final de materia Análisis Estadístico Multivariado                               -- #
# -- Codigo: principal.py - codigo de flujo principal                                                    -- #
# -- Repositorio: https://github.com/IFFranciscoME/FinTechLab/tree/master/MCD-ITESO/AEM/aem_proyecto     -- #
# -- Autor: Francisco ME                                                                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import funciones as fn                          # Importar funcione especiales hechas para este proyecto
import visualizaciones as vs                    # Importar funciones para visualizaciones
from datos import df_pe_w as df_pe_w            # Importar los precios historicos semanales
from datos import df_ce_w as df_ce_w            # Importar los indicadores económicos historicos
from datos import df_pe_m1 as df_pe_m1          # Importar los indicadores económicos historicos

# -- ---------------------------------------------------------------------------- Visualización de datos -- #
# -- visualizacion de los primero 50 precios OHLC semanales
g1 = vs.g_velas(p0_de=df_pe_w[1:50])

# -- visualizacion de los primero 50 precios OHLC diarios para mostrar reaccion del precio
g2 = vs.g_velas_reaccion(p0_de=df_pe_m1[0:29])

# -- SLIDES visualizacion de precios semanales
slides_df1 = df_pe_w.copy().iloc[1:5, :]

# -- SLIDES visualizacion de precios semanales
slides_df2 = df_pe_m1.copy().iloc[1:5, :]

# -- SLIDES visualizacion de calendario económico
slides_df3 = df_ce_w.copy().iloc[27:32, :-1]

# -- --------------------------------------------------------------------------- Ingeniería de variables -- #
# -- generacion de variables endogenas
df1 = fn.f_features_end(p_datos=df_pe_w)

# -- SLIDES visualizacion de variables endogenas - timestamp y variables dependientes
slides_df4 = df1.copy()[['co', 'ho', 'ol']][0:3]

# -- SLIDES visualizacion de variables endogenas - timestamp y variables independientes
slides_df5 = df1.copy()[['co', 'lag_ho_1', 'ma_ho_2', 'lag_ol_1', 'ma_ol_2',
                         'lag_ho_2', 'ma_ho_3', 'lag_ol_2', 'ma_ol_3']][0:3]

# -- ------------------------------------------------------------------------- Modelo 1 y Modelo 2 - pca -- #
# -- ajute de modelo 1A: RLM con variables endogenas (sin tratamiento)
m1 = fn.f_rlm(p_datos=df1, p_y='co')

# -- utilizar PCA para reducir dimensionalidad de modelo 1
df2 = fn.f_pca(p_datos=df1, p_exp=0.85)

# -- ajuste de modelo 1B: RLM con variables endogenas (reducido con PCA)
m2 = fn.f_rlm(p_datos=df2, p_y='pca_y')

# -- --------------------------------------------------------------------- Modelo 3: ANOVA para reacción -- #

# Para saber si es conveniente utilizar la comunicación de indicadores economicos
# como "gatillo" que genera los patrones en las series de tiempo.

m4 = fn.f_anova(p_datos_ce=df_ce_w, p_datos_ph=df_pe_m1)
df3 = df_ce_w.copy()[3:8]
df4 = m4

# regresa data frame con indicadores y escenarios que pasaron prueba de anova = no hay diferencia entre
# las reacciones del precio en el escenario indicado en el dataframe de resultado.

# -- --------------------------------------------------------------------------------- Modelo 4 STS-MASS -- #

# -- para cada indicador, en su escenario detectado por anova, buscar si hubo repetición de patrones en
# por lo menos, 50% de los casos.

# USD Initial Jobless Claims
# -- 2009-02-05 13:30:00+00:00 --> '2015-07-02 12:30:00+0000'
# -- 2009-05-21 12:30:00+00:00 --> '2013-02-14 13:30:00+0000'
# -- 2009-02-05 13:30:00+00:00 --> '2015-07-02 12:30:00+0000'
# -- 2012-04-19 12:30:00+00:00 --> '2019-08-22 12:30:00+0000'
# -- 2013-08-29 12:30:00+00:00 --> '2019-06-20 12:30:00+0000'
# -- 2013-11-27 13:30:00+00:00 --> '2017-09-14 12:30:00+0000'
# -- 2016-11-17 13:30:00+00:00 --> '2012-12-06 13:30:00+0000'
# -- 2017-03-02 13:30:00+00:00 --> '2016-09-22 12:30:00+0000'
# -- 2017-11-16 13:30:00+00:00 --> '2013-06-27 12:30:00+0000'

# df_ce_w_1 = df_ce_w.copy()[df_ce_w['Name'] == 'USD Initial Jobless Claims']
# m5 = fn.f_stsc_mass(p_precios=df_pe_m1, p_calendario=df_ce_w_1, p_indicadores=m4, p_ventana=30)
