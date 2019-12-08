
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Proyecto: Proyecto final de materia Análisis Estadístico Multivariado                               -- #
# -- Codigo: principal.py - codigo de flujo principal                                                    -- #
# -- Repositorio: https://github.com/IFFranciscoME/FinTechLab/tree/master/MCD-ITESO/AEM/aem_proyecto     -- #
# -- Autor: Francisco ME                                                                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import funciones as fn                          # Importar funcione especiales hechas para este proyecto
# import visualizaciones as vs                    # Importar funciones para visualizaciones
# from datos import df_pe_w as df_pe_w            # Importar los precios historicos semanales
from datos import df_ce_w as df_ce_w            # Importar los indicadores económicos historicos
from datos import df_pe_m1 as df_pe_m1          # Importar los indicadores económicos historicos

# -- -------------------------------------------------------------------------------------- Introduccion -- #
# -- contexto, definiciones

# -- ------------------------------------------------------------------------------- Problema a resolver -- #
# -- explicacion de algoritmo

# -- ----------------------------------------------------------------------------- Aplicacion de trading -- #
# -- explicacion de conceptos de trading y contextualizacion de problema

# -- -------------------------------------------------------------------------------- Obtencion de datos -- #
# -- explicar datos.py, df_precios_w, df_precios_m5, df_ce_w

# -- ---------------------------------------------------------------------------- Visualización de datos -- #
# -- explicar visualizaciones.py

# -- visualizacion de precios semanales
# grafica1 = vs.g_velas(p0_de=df_pe_w)
#
# -- --------------------------------------------------------------------------- Ingeniería de variables -- #
# # -- explicación de funciones.py
#
# # -- generacion de variables endogenas
# df_datos_end = fn.f_features_end(p_datos=df_pe_w)
#
# -- ------------------------------------------------------------------------- Modelo 1 y Modelo 2 - pca -- #
#
# # -- ajute de modelo 1A: RLM con variables endogenas (sin tratamiento)
# res1 = fn.f_rlm(p_datos=df_datos_end, p_y='co')
#
# # -- utilizar PCA para reducir dimensionalidad de modelo 1
# df_pca = fn.f_pca(p_datos=df_datos_end, p_exp=0.80)
#
# # -- ajuste de modelo 1B: RLM con variables endogenas (reducido con PCA)
# res2 = fn.f_rlm(p_datos=df_pca, p_y='pca_y')
#
# # -- comparativa de resultados entre modelos
# res3 = []

# -- --------------------------------------------------------------------- Modelo 3: ANOVA para reacción -- #

# Para saber si es conveniente utilizar la comunicación de indicadores economicos
# como "gatillo" que genera los patrones en las series de tiempo.

res3_anova = fn.f_anova(p_datos_ce=df_ce_w, p_datos_ph=df_pe_m1)

# regresa data frame con indicadores y escenarios que pasaron prueba de anova = no hay diferencia entre
# las reacciones del precio en el escenario indicado en el dataframe de resultado.

# -- --------------------------------------------------------------------------------- Modelo 4 STS-MASS -- #

# -- para cada indicador, en su escenario detectado por anova, buscar si hubo repetición de patrones en
# por lo menos, 50% de los casos.
fn.f_stsc_mass(p_precios=df_pe_m1, p_calendario=df_ce_w, p_indicadores=res3_anova, p_ventana=30)

# -- ----------------------------------------------------------------------------- Estrategia de trading -- #

# -- armar codigo generico para estrategia de trading
# -- aplicar a caso RLM-END
# -- aplicar a caso STS-MASS

# -- ------------------------------------------------------------------ Modelo 5 Backtest con clustering -- #
# -- Kmeans para Clustering de configuraciones tp/sl (Destacar que no son series de tiempo)

# -- ---------------------------------------------------------------------------- Análisis de resultados -- #

# -- ---------------------------------------------------------------------------------- Trabajo a futuro -- #
