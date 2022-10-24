
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: main.py : python script with the main functionality                                         -- #
# -- author: team 0                                                                                      -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://deepnote.com/workspace/pgomezman-fb44-564aea80-b395-487c-9bd5-0c594e3cbf9f/project/MyST-Lab3-6769db9b-c501-49f1-b3ff-df4df228bebc/%2Ffunctions.py                                                                    -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# Librerias y dependencias
import functions as fn
import visualizations as vs

# Capital
C=100000

# Emiliano
E1=fn.f_leer_archivo('files/reporteEmiliano.csv')
E2=fn.f_columnas_tiempos(E1)
E3=fn.f_columnas_pips(E2)
E4=fn.f_estadisticas_ba(E3)
E5=fn.profit_acm_d(E3)
E6=fn.f_evolucion_capital(E3)
E7=fn.f_estadisticas_mad(0.05,E6,E1)
EG1=vs.fn_graph_rank(E4['df_2_ranking'])
EG2=vs.fn_capital_plot(E6,E7,C)

# Luis
L1=fn.f_leer_archivo('files/reporteLuis.csv')
L2=fn.f_columnas_tiempos(L1)
L3=fn.f_columnas_pips(L2)
L4=fn.f_estadisticas_ba(L3)
L5=fn.profit_acm_d(L3)
L6=fn.f_evolucion_capital(L3)
L7=fn.f_estadisticas_mad(0.05,L6,L1)
LG1=vs.fn_graph_rank(L4['df_2_ranking'])
LG2=vs.fn_capital_plot(L6,L7,C)

# Pao
P1=fn.f_leer_archivo('files/reportePao1.csv')
P2=fn.f_columnas_tiempos(P1)
P3=fn.f_columnas_pips(P2)
P4=fn.f_estadisticas_ba(P3)
P5=fn.profit_acm_d(P3)
P6=fn.f_evolucion_capital(P3)
P7=fn.f_estadisticas_madP(0.05,P6,P1)
PG1=vs.fn_graph_rank(P4['df_2_ranking'])
PG2=vs.fn_capital_plot(P6,P7,C)

# Xavier
X1=fn.f_leer_archivo('files/reporteXavier.csv')
X2=fn.f_columnas_tiempos(X1)
X3=fn.f_columnas_pips(X2)
X4=fn.f_estadisticas_ba(X3)
X5=fn.profit_acm_d(X3)
X6=fn.f_evolucion_capital(X3)
X7=fn.f_estadisticas_madX(0.05,X6,X1)
XG1=vs.fn_graph_rank(X4['df_2_ranking'])
XG2=vs.fn_capital_plot(X6,X7,C)