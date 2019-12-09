
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Proyecto: Proyecto final de materia Análisis Estadístico Multivariado                               -- #
# -- Codigo: visualizaciones.py - codigos para generar visualizaciones para el proyecto                  -- #
# -- Repositorio: https://github.com/IFFranciscoME/FinTechLab/tree/master/MCD-ITESO/AEM/aem_proyecto     -- #
# -- Autor: Francisco ME                                                                                 -- #
# -- --------------------------------------------------------------------------------------------------- -- #

import plotly.graph_objs as go                            # objetos de imagenes para funcion principal
import plotly.io as pio                                   # renderizador para visualizar imagenes
import numpy as np                                        # funciones numericas
pio.renderers.default = "browser"                         # render de imagenes para correr en script


# -- ------------------------------------------------------------------------ GRÁFICA: velas OHLC Simple -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_velas(p0_de):
    """
    :param p0_de: data frame con datos a graficar
    :return fig:

    p0_de = datos_dd
    p1_pa = 'sell'
    datos_dd = pd.DataFrame({'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': []}, index=[])
    """

    p0_de.columns = [list(p0_de.columns)[i].lower() for i in range(0, len(p0_de.columns))]

    fig = go.Figure(data=[go.Candlestick(x=p0_de['timestamp'],
                                         open=p0_de['open'], high=p0_de['high'],
                                         low=p0_de['low'], close=p0_de['close'])])

    fig.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=0),
                      title=dict(x=0.5, y=1, text='Precios Historicos OHLC'),
                      xaxis=dict(title_text='Hora del dia', rangeslider=dict(visible=False)),
                      yaxis=dict(title_text='Precio del EurUsd'))

    fig.layout.autosize = False
    fig.layout.width = 840
    fig.layout.height = 520

    return fig


# -- ---------------------------------------------------------------------- GRÁFICA: velas OHLC Reaccion -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_velas_reaccion(p0_de):
    """
    :param p0_de: data frame con datos a graficar
    :return fig:

    p0_de = datos_dd
    p1_pa = 'sell'
    datos_dd = pd.DataFrame({'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': []}, index=[])
    """

    # p0_de['timestamp'] = [(pd.to_datetime(p0_de['timestamp']))[x].tz_localize('UTC')
    #                           for x in range(0, len(p0_de['timestamp']))]

    f_i = p0_de['timestamp'].loc[0]

    yini = p0_de['high'][0]
    yfin = max(p0_de['high'])
    fig = go.Figure(data=[go.Candlestick(x=p0_de['timestamp'],
                                         open=p0_de['open'], high=p0_de['high'],
                                         low=p0_de['low'], close=p0_de['close'])])

    lineas = [dict(x0=f_i, x1=f_i, xref='x', y0=yini, y1=yfin, yref='y', type='line',
                   line=dict(color='red', width=1.5, dash='dashdot'))]

    fig.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=0),
                      title=dict(x=0.5, text='Reacción del precio (intradía) ante el comunicado de un'
                                             ' <b> indicador económico </b>'),
                      xaxis=dict(title_text='Hora del dia', rangeslider=dict(visible=False)),
                      yaxis=dict(title_text='Precio del EurUsd'),
                      annotations=[go.layout.Annotation(x=f_i, y=1.025, xref="x",
                                                        yref="paper", showarrow=False,
                                                        text="Indicador Comunicado")],
                      shapes=lineas)

    fig.layout.autosize = False
    fig.layout.width = 840
    fig.layout.height = 520

    return fig


# -- ------------------------------------------------------------------ GRÁFICA: lineas series de tiempo -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def g_lineas(p_datos1, p_datos2):
    """
    :param p_datos1
    :param p_datos2

    :return:
    """

    n = len(p_datos1)
    random_x = list(np.arange(n))
    random_y0 = p_datos1
    random_y1 = p_datos2

    # Create traces
    fig = go.Figure()

    fig.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=50, pad=0),
                      title=dict(x=0.5, y=1, text='Patron encontrado'),
                      xaxis=dict(title_text='fechas', rangeslider=dict(visible=False)),
                      yaxis=dict(title_text='precios (co)'))

    fig.add_trace(go.Scatter(x=random_x, y=random_y0, mode='lines', name='serie_query'))
    fig.add_trace(go.Scatter(x=random_x, y=random_y1, mode='lines', name='serie'))

    fig.update_layout(legend_orientation="h")

    fig.layout.autosize = False
    fig.layout.width = 840
    fig.layout.height = 520

    return fig
