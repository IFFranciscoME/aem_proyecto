
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
# from datos import df_pe_m1 as df_pe_m1          # Importar los indicadores económicos historicos

# -- ---------------------------------------------------------------------------- Visualización de datos -- #
# -- visualizacion de los primero 50 precios OHLC semanales
g1 = vs.g_velas(p0_de=df_pe_w[1:50])

# -- SLIDES visualizacion de precios semanales
slides_df1 = df_pe_w.copy().iloc[1:3, :]

# -- SLIDES visualizacion de precios semanales
slides_df2 = df_pe_m1.copy().iloc[1:3, :]

# -- --------------------------------------------------------------------------- Ingeniería de variables -- #
# -- generacion de variables endogenas
df1 = fn.f_features_end(p_datos=df_pe_w)

# -- SLIDES visualizacion de variables endogenas - timestamp y variables dependientes
slides_df3 = df1.copy()[['co', 'ho', 'ol']][0:3]

# -- SLIDES visualizacion de variables endogenas - timestamp y variables independientes
slides_df4 = df1.copy()[['co', 'lag_ho_1', 'ma_ho_2', 'lag_ol_1', 'ma_ol_2']][0:3]

# -- ------------------------------------------------------------------------- Modelo 1 y Modelo 2 - pca -- #
# -- ajute de modelo 1A: RLM con variables endogenas (sin tratamiento)
m1 = fn.f_rlm(p_datos=df1, p_y='co')

# -- utilizar PCA para reducir dimensionalidad de modelo 1
df2 = fn.f_pca(p_datos=df1, p_exp=0.85)

# -- ajuste de modelo 1B: RLM con variables endogenas (reducido con PCA)
m2 = fn.f_rlm(p_datos=df2, p_y='pca_y')

# # -- comparativa de resultados entre modelos
df_m1m2 = []

# -- --------------------------------------------------------------------- Modelo 3: ANOVA para reacción -- #

# Para saber si es conveniente utilizar la comunicación de indicadores economicos
# como "gatillo" que genera los patrones en las series de tiempo.

# m4 = fn.f_anova(p_datos_ce=df_ce_w, p_datos_ph=df_pe_m1)

# regresa data frame con indicadores y escenarios que pasaron prueba de anova = no hay diferencia entre
# las reacciones del precio en el escenario indicado en el dataframe de resultado.

# -- --------------------------------------------------------------------------------- Modelo 4 STS-MASS -- #

# -- para cada indicador, en su escenario detectado por anova, buscar si hubo repetición de patrones en
# por lo menos, 50% de los casos.
# m5 = fn.f_stsc_mass(p_precios=df_pe_m1, p_calendario=df_ce_w, p_indicadores=m4, p_ventana=30)
