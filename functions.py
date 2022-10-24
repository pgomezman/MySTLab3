
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Functions for lab 3 of MyST                                                                -- #
# -- script: functions.py : python script with data functions                                            -- #
# -- authors: team 0                                                                                     -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://deepnote.com/workspace/pgomezman-fb44-564aea80-b395-487c-9bd5-0c594e3cbf9f/project/MyST-Lab3-6769db9b-c501-49f1-b3ff-df4df228bebc/%2Ffunctions.py                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
# LABORATORIO 3 EQUIPO 0

# Librerias y dependencias
import MetaTrader5 as mt5
import sys
import subprocess
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import functions as fn
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'yfinance'])

# --------------- TRATAMIENTO DE LOS DATOS -------------------

# Estadística Descriptiva

## f_leer_archivo
def f_leer_archivo(param_archivo):
    archivo=pd.read_csv(param_archivo)
    archivo=archivo.fillna(0)
    return archivo

## f_pip_size
def f_pip_size(param_ins):
    df_pips=pd.read_csv('files/instruments_pips.csv')
    df_pips['Instrument']=df_pips['Instrument'].replace({'_':''},regex=True)
    if any(param_ins==df_pips['Instrument']):
        pips=(1/df_pips[param_ins==df_pips['Instrument']]['TickSize']).iloc[0]
    else:
        pips=100
    return pips

## f_columnas_tiempos
def f_columnas_tiempos(param_data):
    param_data['Open Time']=pd.to_datetime(param_data['Open Time'])
    param_data['Close Time']=pd.to_datetime(param_data['Close Time'])
    param_data['Tiempo']=(param_data['Close Time']-param_data['Open Time']).astype('timedelta64[s]')
    return param_data

## f_columnas_pips
def f_columnas_pips(param_data):
    pipmult=param_data['Symbol'].apply(f_pip_size)
    param_data['Pips']=np.where(param_data['Type']=='buy',(param_data['Close Price']-param_data['Open Price'])*pipmult,(param_data['Open Price']-param_data['Close Price'])*pipmult)
    param_data['Pips']=param_data['Pips'].round(0)
    param_data['Pips_acm']=param_data['Pips'].cumsum()
    param_data['Profit_acm']=param_data['Profit'].cumsum().round(2)
    return param_data

## f_estadisticas_ba
def f_estadisticas_ba(param_data):
    OT=param_data['Symbol'].count()
    OG=param_data[param_data['Profit']>0]['Profit'].count()
    OGC=param_data[(param_data['Type']=='buy')&(param_data['Profit']>0)]['Profit'].count()
    OGV=param_data[(param_data['Type']=='sell')&(param_data['Profit']>0)]['Profit'].count()
    OP=param_data[param_data['Profit']<0]['Profit'].count()
    OPC=param_data[(param_data['Type']=='buy')&(param_data['Profit']<0)]['Profit'].count()
    OPV=param_data[(param_data['Type']=='sell')&(param_data['Profit']<0)]['Profit'].count()
    M=param_data['Profit'].median()
    MP=param_data['Pips'].median()
    RE=OG/OT
    RP=OG/OP
    REC=OGC/OT
    REV=OGV/OT
    df_1_tabla=pd.DataFrame({
        'Medida':['Ops totales','Ganadoras','Ganadoras_c','Ganadoras_v','Perdedoras','Perdedoras_c',
        'Perdedoras_v','Mediana (Profit)','Mediana (Pips)','R_efectividad','R_proporcion','R_efectividad_c',
        'R_efectividad_v'],
        'Valor':[OT, OG, OGC, OGV, OP, OPC, OPV, M, MP, RE, RP, REC, REV],
        'Descripcion':['Operaciones totales','Operaciones ganadoras','Operaciones ganadoras de compra',
        'Operaciones ganadoras de venta','Operaciones perdedoras','Operaciones perdedoras de compra',
        'Operaciones perdedoras de venta','Mediana de profit de operaciones','Mediana de pips de operaciones',
        'Ganadoras Totales/Operaciones Totales','Ganadoras Totales/Perdedoras Totales',
        'Ganadoras Compras/Operaciones Totales','Ganadoras Ventas/Operaciones Totales']
    })
    tabla_2=[]
    symbol=param_data['Symbol'].unique()
    for i in symbol:
        a=param_data[param_data['Symbol']==i]['Symbol'].count()
        b=param_data[(param_data['Profit']>0)&(param_data['Symbol']==i)]['Profit'].count()
        tabla_2.append({
            'Symbol':i,
            'Rank':((b/a)*100).round(2)
        })
    df_2_ranking=pd.DataFrame(tabla_2)
    diccionario={'df_1_tabla':df_1_tabla,'df_2_ranking':df_2_ranking}
    return diccionario

# Métricas de Atribución al Desempeño

## profit_acm_d
def profit_acm_d(param_data):
    capital=100000
    param_data['Profit_acm_d']=capital+param_data['Profit_acm']
    return param_data


## f_evolucion_capital
def f_evolucion_capital(param_data):
    capital=100000
    df_3=pd.DataFrame({
        'Timestamp':param_data['Open Time'].dt.strftime('%Y-%m-%d').unique(),
        'Profit_d':param_data.groupby(param_data['Open Time'].dt.strftime('%Y-%m-%d'))['Profit'].sum().values.round(2),
    })
    df_3['Profit_acm_d']=capital+df_3['Profit_d'].cumsum().round(2)
    return df_3


## Descargar datos de yahoo para el benchmark
def adjP(tickers):
    end = "2022-9-27"
    start = "2022-9-20"
    stockPrices = (yf.download(tickers, start = start, end = end)["Adj Close"]).dropna()
    return stockPrices

## f_estadisticas_mad
def f_estadisticas_mad(rf,df4,param_data):

    dfdf = param_data
    
    #Sharpe_ratio
    rp = np.log(df4.Profit_acm_d) - np.log(df4.Profit_acm_d.shift(periods=1)) #rendimientos logarimicos
    rp[0]=0
    sharp_original= (rp.mean() - (rf/252)) / rp.std()

    #Sharpe_Actualizado
    start=df4['Timestamp'].iloc[0]
    end=(datetime.strptime(df4['Timestamp'].iloc[-1], '%Y-%m-%d')+timedelta(days=1)).strftime('%Y-%m-%d')
    sp500=(yf.download(tickers='^GSPC', start=start, end=end, interval='1d')['Adj Close']).dropna()
    rp500=np.log(sp500) - np.log(sp500.shift(periods=1)) #rendimientos logaritmicos
    rp500[0]=0
    sharpe_actualizado = (rp.mean() - rp500.mean()) / (rp.values-rp500.values).std()

    # Drawdown & Drawup 
    min_max_df = pd.DataFrame(index=range(2), columns=['Open_Time', 'Close_Time'])
    for i in range(len(dfdf)):
        if dfdf['Profit'][i] == dfdf['Profit'].min(): #Se guarda en la fila 0 el tiempo en donde se registro el valor flotante mas pequeño
            min_max_df['Open_Time'][0] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][0] = dfdf['Close Time'][i]
        elif dfdf['Profit'][i] == dfdf['Profit'].max(): #Se guarda en la fila 1 el tiempo en donde se registro el valor flotante mas grande
            min_max_df['Open_Time'][1] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][1] = dfdf['Close Time'][i]

    #DataFrame de f_estadisticas_mad
    df_estadisticas_mad = pd.DataFrame(index=range(8), columns=['Métrica', ' ', 'Valor', 'Descripción'])
    df_estadisticas_mad['Métrica'] = ['sharp_original','sharpe_actualizado','drawdown_capi','drawdown_capi','drawdown_capi','drawup_capi','drawup_capi','drawup_capi']
    df_estadisticas_mad[' '] = ['Cantidad','Cantidad','Fecha Inicial','Fecha Final','DrawDown $ (capital)','Fecha Inicial','Fecha Final','DrawUp $ (capital)']
    df_estadisticas_mad['Descripción'] = ['Sharpe Ratio Fórmula Original','Sharpe Ratio Fórmula Original','Fecha inicial del DrawDown de Capital','Fecha final del DrawDown de Capital','Máxima pérdida flotante registrada','Fecha inicial del DrawUp de Capital','	Fecha final del DrawUp de Capital','Máxima ganancia flotante registrada']
    df_estadisticas_mad['Valor'] = [round(sharp_original,2),round(sharpe_actualizado,2),min_max_df['Open_Time'][0],min_max_df['Close_Time'][0],dfdf['Profit'].min(),min_max_df['Open_Time'][1],min_max_df['Close_Time'][1],dfdf['Profit'].max()]

    return df_estadisticas_mad


## f_estadisticas_mad Paola
def f_estadisticas_madP(rf,df4,param_data):

    dfdf = param_data
    
    #Sharpe_ratio
    rp = np.log(df4.Profit_acm_d) - np.log(df4.Profit_acm_d.shift(periods=1)) #rendimientos logarimicos
    rp[0]=0
    sharp_original= (rp.mean() - (rf/252)) / rp.std()

    #Sharpe_Actualizado
    start=df4['Timestamp'].iloc[0]
    end=(datetime.strptime(df4['Timestamp'].iloc[-1], '%Y-%m-%d')+timedelta(days=1)).strftime('%Y-%m-%d')
    sp500=(yf.download(tickers='^GSPC', start=start, end=end, interval='1d')['Adj Close']).dropna()
    #rp500=np.log(sp500) - np.log(sp500.shift(periods=1)) #rendimientos logaritmicos
    #rp500[0]=0
    # Paola opero los fines de semana entonces cambia su rp500
    rp500=np.array([ 0.0,0.00342268634953236 ,0.00341101151562029 ,-0.0113361147515754 ,-0.017264674096884 ,-0.00846329398764389 ,-0.0173828288318845 ,-0.00518367510902351 ,-0.0103943607804595])

    sharpe_actualizado = (rp.mean() - rp500.mean()) / (rp.values-rp500).std()

    # Drawdown & Drawup 
    min_max_df = pd.DataFrame(index=range(2), columns=['Open_Time', 'Close_Time'])
    for i in range(len(dfdf)):
        if dfdf['Profit'][i] == dfdf['Profit'].min(): #Se guarda en la fila 0 el tiempo en donde se registro el valor flotante mas pequeño
            min_max_df['Open_Time'][0] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][0] = dfdf['Close Time'][i]
        elif dfdf['Profit'][i] == dfdf['Profit'].max(): #Se guarda en la fila 1 el tiempo en donde se registro el valor flotante mas grande
            min_max_df['Open_Time'][1] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][1] = dfdf['Close Time'][i]

    #DataFrame de f_estadisticas_mad
    df_estadisticas_mad = pd.DataFrame(index=range(8), columns=['Métrica', ' ', 'Valor', 'Descripción'])
    df_estadisticas_mad['Métrica'] = ['sharp_original','sharpe_actualizado','drawdown_capi','drawdown_capi','drawdown_capi','drawup_capi','drawup_capi','drawup_capi']
    df_estadisticas_mad[' '] = ['Cantidad','Cantidad','Fecha Inicial','Fecha Final','DrawDown $ (capital)','Fecha Inicial','Fecha Final','DrawUp $ (capital)']
    df_estadisticas_mad['Descripción'] = ['Sharpe Ratio Fórmula Original','Sharpe Ratio Fórmula Original','Fecha inicial del DrawDown de Capital','Fecha final del DrawDown de Capital','Máxima pérdida flotante registrada','Fecha inicial del DrawUp de Capital','	Fecha final del DrawUp de Capital','Máxima ganancia flotante registrada']
    df_estadisticas_mad['Valor'] = [round(sharp_original,2),round(sharpe_actualizado,2),min_max_df['Open_Time'][0],min_max_df['Close_Time'][0],dfdf['Profit'].min(),min_max_df['Open_Time'][1],min_max_df['Close_Time'][1],dfdf['Profit'].max()]

    return df_estadisticas_mad


## f_estadisticas_mad
def f_estadisticas_madX(rf,df4,param_data):

    dfdf = param_data
    
    #Sharpe_ratio
    rp = df4.Profit_acm_d.pct_change() #rendimientos con change por capital negativo
    rp[0]=0
    sharp_original= (rp.mean() - (rf/252)) / rp.std()

    #Sharpe_Actualizado
    start=df4['Timestamp'].iloc[0]
    end=(datetime.strptime(df4['Timestamp'].iloc[-1], '%Y-%m-%d')+timedelta(days=1)).strftime('%Y-%m-%d')
    sp500=(yf.download(tickers='^GSPC', start=start, end=end, interval='1d')['Adj Close']).dropna()
    #rp500=np.log(sp500) - np.log(sp500.shift(periods=1)) #rendimientos logaritmicos
    #rp500[0]=0
    # Xavier no opero todos los dias
    rp500=np.array([0 ,-0.017265 ,-0.008463 ,-0.017383 ,-0.010394 ,-0.028403])
    sharpe_actualizado = (rp.mean() - rp500.mean()) / (rp.values-rp500).std()

    # Drawdown & Drawup 
    min_max_df = pd.DataFrame(index=range(2), columns=['Open_Time', 'Close_Time'])
    for i in range(len(dfdf)):
        if dfdf['Profit'][i] == dfdf['Profit'].min(): #Se guarda en la fila 0 el tiempo en donde se registro el valor flotante mas pequeño
            min_max_df['Open_Time'][0] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][0] = dfdf['Close Time'][i]
        elif dfdf['Profit'][i] == dfdf['Profit'].max(): #Se guarda en la fila 1 el tiempo en donde se registro el valor flotante mas grande
            min_max_df['Open_Time'][1] = dfdf['Open Time'][i]
            min_max_df['Close_Time'][1] = dfdf['Close Time'][i]

    #DataFrame de f_estadisticas_mad
    df_estadisticas_mad = pd.DataFrame(index=range(8), columns=['Métrica', ' ', 'Valor', 'Descripción'])
    df_estadisticas_mad['Métrica'] = ['sharp_original','sharpe_actualizado','drawdown_capi','drawdown_capi','drawdown_capi','drawup_capi','drawup_capi','drawup_capi']
    df_estadisticas_mad[' '] = ['Cantidad','Cantidad','Fecha Inicial','Fecha Final','DrawDown $ (capital)','Fecha Inicial','Fecha Final','DrawUp $ (capital)']
    df_estadisticas_mad['Descripción'] = ['Sharpe Ratio Fórmula Original','Sharpe Ratio Fórmula Original','Fecha inicial del DrawDown de Capital','Fecha final del DrawDown de Capital','Máxima pérdida flotante registrada','Fecha inicial del DrawUp de Capital','	Fecha final del DrawUp de Capital','Máxima ganancia flotante registrada']
    df_estadisticas_mad['Valor'] = [round(sharp_original,2),round(sharpe_actualizado,2),min_max_df['Open_Time'][0],min_max_df['Close_Time'][0],dfdf['Profit'].min(),min_max_df['Open_Time'][1],min_max_df['Close_Time'][1],dfdf['Profit'].max()]

    return df_estadisticas_mad


def f_be_de(param_data):   
    dfdf=param_data
    df=param_data
    df2=f_columnas_tiempos(df)
    df3=f_columnas_pips(df2)
    mt5.initialize(
    #  path,                     // path to the MetaTrader 5 terminal EXE file
       login=5501309,              #// account number
       password="iZqfSq3L",      #// password
       server="FxPro-MT5",          #// server name as it is specified in the terminal
    #  timeout=TIMEOUT,          #// timeout
    #  portable=False            #// portable mode
       )



    anclas = df3[df3['Profit'] > 0]
    valor = -1 
    li=[]
    for i in range(len(anclas)):
        valor= valor + 1
        li.append(valor)
    anclas.index = pd.Series(li)  


    test = pd.DataFrame([])
    index_ancla = []
    for i in range(len(anclas)):
        test1 = df3[(df3['Close Time'] > anclas['Close Time'][i]) & (df3['Open Time'] > anclas['Open Time'][i]) & (df3['Open Time'] < anclas['Close Time'][i])]
        test1['index_ancla'] = i
        test1['time_ancla'] = anclas['Close Time'][i]
        test = pd.concat([test1,test])


    valor = -1 
    li=[]
    for i in range(len(test)):
        valor= valor + 1
        li.append(valor)
    test.index = pd.Series(li)  


    sesgo = []
    bandera = 0
    sesgo_gen = 0


    for i in range(len(test)):
        symbol = mt5.copy_rates_range(test['Symbol'][i], mt5.TIMEFRAME_M1, pd.to_datetime(test['time_ancla'])[i] - timedelta(.0009),pd.to_datetime(test['time_ancla'])[i])
        symbol_price = symbol[0][4]
        if (test['Type'][i] == 'buy') or (test['Type'][i] == 'sell'):
            sesgo.append((test['Open Price'][i] - symbol_price)*test['Volume'][i] * fn.f_pip_size(test['Symbol'][i]))
            #test['sesgo'][i] = sesgo[i]

    num_sesgo = len(sesgo)

    for i in range(len(sesgo)):
        if sesgo[i] == min(sesgo):
            min_num = i

    for i in range(len(sesgo)):
        if sesgo[i] == max(sesgo):
            max_num = i


    quo = []
    contador = 0
    for i in range(len(anclas)):
        status = sesgo < anclas['Profit'][i]
        quo.append(status)
        for j in range(len(status)):
            if status[j] == True:
                contador = contador + 1

    status_quo = contador / (len(quo)*5)

    aver = []
    contador = 0
    for i in range(len(anclas)):
        aversion = (abs(sesgo / anclas['Profit'][i]) > 2)
        aver.append(aversion)
        for j in range(len(aversion)):
            if aversion[j] == True:
                contador = contador + 1

    aversion_perdida = contador / (len(aver)*5)

    era = []
    contador = 0
    for i in range(len(anclas)):
        terc = abs(sesgo / anclas['Profit_acm'][i])
        era.append(terc)


    primera = df3['Profit_acm'].iloc[-1] > 0

    segunda = (sesgo[-1] > sesgo[0]) & (anclas['Profit'].iloc[-1] > anclas['Profit'].iloc[0])

    tercera = (np.mean(era) / anclas['Profit'].iloc[-1]) > 2

    if ((primera == True) & (segunda == True)) or ((primera == True)&(tercera == True)) or ((segunda == True)&(tercera == True)):
        des = 'Si'
    else:
        des = 'No'

    resultados_dic = pd.DataFrame(index = range(1), columns = ['ocurrencias', 'status_quo', 'aversion_perdida', 'sensibilidad_decreciente'])
    resultados_dic['ocurrencias'][0] = num_sesgo
    resultados_dic['status_quo'][0] = status_quo 
    resultados_dic['aversion_perdida'][0] = aversion_perdida
    resultados_dic['sensibilidad_decreciente'][0] = des

    ocurrencias = {}
    ocurrencias['Cantidad'] = num_sesgo
    for i in range(num_sesgo):
        ocurrencia = f'ocurrencia_{i+1}'
        ocurrencias[ocurrencia]={'timestamp':{test['time_ancla'][i]},'operaciones':{}}
        valor = 'ganandora' if (sesgo[i] > 0) else 'perdedora'
        valor_1 = 'profit_ganandora' if sesgo[i] > 0 else 'profit_perdedora'
        ocurrencias[ocurrencia]['operaciones'][valor]={'instrumento':{test['Symbol'][i]},'volumen':{test['Volume'][i]},'sentido':{test['Type'][i]},valor_1:{sesgo[i]}}
    ocurrencias['ratio_cp_profit_acm'] = min(sesgo) / (test['Profit_acm'][min_num]+100000)
    ocurrencias['ratio_cg_profit_acm'] = max(sesgo) / (test['Profit_acm'][max_num]+100000)
    ocurrencias['ratio_cp_cg'] = min(sesgo) / max(sesgo)

    ocurrencias['resultados'] = resultados_dic
    
    return ocurrencias