import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import warnings
import os

warnings.filterwarnings('ignore')

URL_STATION_INFO = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_information.json"
URL_STATION_STATUS = "https://gbfs.mex.lyftbikes.com/gbfs/es/station_status.json"

DISTANCIA_LIMITE_GRADOS = 0.009

FEATURES_REGRESOR = [
    'capacity', 'bikes_avail', 'pct_full', 'hora_sin', 'hora_cos', 'dia_semana',
    'es_fin_de_semana', 'hora_tipo_dia', 'es_pico_semana', 'pct_full_log1', 'pct_full_log2', 'pct_full_log3',
    'pct_full_log_1h', 'rango_historico_estacion', 'velocidad_cambio',
    'aceleracion_cambio', 'tendencia_1h', 'ocupacion_historica_estacion_hora',
    'slots_vacios', 'zona_logistica', 'rolling_mean_30m', 'rolling_std_30m', 'rolling_mean_1h',
    'interaccion_cap_hora', 'historico_estacion_hora_dia', 'desviacion_de_media_30m', 'momentum_quiebre_30m'
]

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
BUFFER_HISTORICO_PATH = os.path.join(ML_DIR, "buffer_tiempo_real.csv")


def cargar_modelos():
    """
    Carga en memoria los archivos serializados de XGBoost (.json) y los DataFrames 
    de calibración histórica generados en la etapa de entrenamiento.
    """
    print("[+] Cargando modelos y mapas base...")
    model_regresor = xgb.XGBRegressor()
    model_regresor.load_model(os.path.join(ML_DIR, "regresor_ecobici.json"))

    model_clasificador = xgb.XGBClassifier()
    model_clasificador.load_model(os.path.join(ML_DIR, "clasificador_ecobici.json"))

    mapa_tri_interaccion = pd.read_csv(os.path.join(ML_DIR, "mapa_tri_interaccion.csv"))
    mapa_historico = pd.read_csv(os.path.join(ML_DIR, "mapa_historico.csv"))
    encoding_map = pd.read_csv(os.path.join(ML_DIR, "encoding_map.csv"))
    coords_base = pd.read_csv(os.path.join(ML_DIR, "coords_base.csv"))
    return model_regresor, model_clasificador, mapa_tri_interaccion, mapa_historico, encoding_map, coords_base


def actualizar_y_obtener_lags(df_actual, ts_actual):
    """
    GESTIÓN DE MEMORIA TEMPORAL (INERCIA):
    Como la API de Ecobici no tiene pasado, esta función mantiene un archivo CSV local 
    como buffer flotante. Filtra datos mayores a 1 hora y extrae las fotos de hace 
    10, 20, 30 y 60 minutos para que el modelo entienda si la estación se está vaciando o llenando.
    """
    df_buffer_nuevo = df_actual[['station_id', 'pct_full']].copy()
    df_buffer_nuevo['run_ts'] = ts_actual

    if os.path.exists(BUFFER_HISTORICO_PATH):
        try:
            df_viejo = pd.read_csv(BUFFER_HISTORICO_PATH)
            df_viejo['run_ts'] = pd.to_datetime(df_viejo['run_ts'])
            
            # Se eliminan los datos con más de una hora de antigüedad
            limite_tiempo = ts_actual - pd.Timedelta(minutes=65)
            df_viejo = df_viejo[df_viejo['run_ts'] >= limite_tiempo]
            
            df_total = pd.concat([df_viejo, df_buffer_nuevo], ignore_index=True)
        except Exception:
            df_total = df_buffer_nuevo
    else:
        df_total = df_buffer_nuevo

    df_total.to_csv(BUFFER_HISTORICO_PATH, index=False)
    df_total = df_total.sort_values(by=['station_id', 'run_ts']).reset_index(drop=True)
    
    resumen_lags = []
    for station_id, grupo in df_total.groupby('station_id'):
        if len(grupo) >= 1:
            pct_actual = grupo['pct_full'].iloc[-1]
            lag1 = grupo['pct_full'].iloc[-2] if len(grupo) >= 2 else pct_actual
            lag2 = grupo['pct_full'].iloc[-3] if len(grupo) >= 3 else lag1
            lag3 = grupo['pct_full'].iloc[-4] if len(grupo) >= 4 else lag2
            lag_1h = grupo['pct_full'].iloc[-6] if len(grupo) >= 6 else lag3
            
            window_30m = grupo['pct_full'].tail(3)
            window_1h = grupo['pct_full'].tail(6)
            
            resumen_lags.append({
                'station_id': station_id,
                'pct_full_log1': lag1, 'pct_full_log2': lag2, 'pct_full_log3': lag3, 'pct_full_log_1h': lag_1h,
                'rolling_mean_30m': window_30m.mean(),
                'rolling_std_30m': window_30m.std() if len(window_30m) > 1 else 0.0,
                'rolling_mean_1h': window_1h.mean()
            })
            
    return pd.DataFrame(resumen_lags)


def ejecutar_prediccion(model_regresor, model_clasificador, mapa_tri_interaccion, mapa_historico, encoding_map, coords_base) -> dict:

    # 1. CONSUMO DESDE ENDPOINTS GBFS
    res_info = requests.get(URL_STATION_INFO, timeout=15).json()
    res_disp = requests.get(URL_STATION_STATUS, timeout=15).json()

    # Procesamiento de metadatos
    df_info_api = pd.DataFrame([{
        'station_id': int(s['station_id']),
        'lat': float(s['lat']),
        'lon': float(s['lon']),
        'capacity': int(s['capacity'])
    } for s in res_info['data']['stations']])

    df_disp_api = pd.DataFrame([{
        'station_id': int(s['station_id']),
        'bikes_avail': int(s['num_bikes_available']),
        'is_renting': int(s['is_renting']),
        'is_returning': int(s['is_returning'])
    } for s in res_disp['data']['stations']])

    # Cruce de inventario con geografía e ingeniería básica de porcentajes de llenado
    df_realtime = pd.merge(df_disp_api, df_info_api, on='station_id', how='inner')
    df_realtime['pct_full'] = df_realtime['bikes_avail'] / df_realtime['capacity'].replace({0: 1})
    df_realtime['slots_vacios'] = df_realtime['capacity'] - df_realtime['bikes_avail']
    df_realtime = df_realtime[(df_realtime['is_renting'] == 1) & (df_realtime['is_returning'] == 1)].reset_index(drop=True)

    # 2. INGENIERÍA DE VARIABLES CRONOLÓGICAS CÍCLICAS
    ts_reciente = pd.Timestamp.now()
    df_realtime['hora'] = ts_reciente.hour
    df_realtime['dia_semana'] = ts_reciente.dayofweek
    df_realtime['es_fin_de_semana'] = df_realtime['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
    df_realtime['hora_sin'] = np.sin(2 * np.pi * df_realtime['hora'] / 24.0)
    df_realtime['hora_cos'] = np.cos(2 * np.pi * df_realtime['hora'] / 24.0)
    df_realtime['hora_tipo_dia'] = df_realtime['hora'] + (df_realtime['es_fin_de_semana'] * 24)
    df_realtime['es_pico_semana'] = df_realtime['dia_semana'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    df_realtime['interaccion_cap_hora'] = df_realtime['capacity'] * df_realtime['hora_cos']

    # 3. MERGE DE TARGET ENCODINGS Y MAPAS GEOGRÁFICOS PRECALCULADOS
    df_realtime = df_realtime.merge(mapa_tri_interaccion, on=['station_id', 'hora', 'dia_semana'], how='left')
    df_realtime = df_realtime.merge(mapa_historico, on=['station_id', 'hora'], how='left')
    df_realtime = df_realtime.merge(encoding_map, on=['station_id', 'hora_tipo_dia'], how='left')
    df_realtime = df_realtime.merge(coords_base, on=['lat', 'lon'], how='left')

    # 4. ACOPLAMIENTO DE INERCIA Y CÁLCULO DE DERIVADAS TEMPORALES
    df_lags = actualizar_y_obtener_lags(df_realtime, ts_reciente)
    df_realtime = df_realtime.merge(df_lags, on='station_id', how='left')

    df_realtime['pct_full_log1'] = df_realtime['pct_full_log1'].fillna(df_realtime['ocupacion_historica_estacion_hora']).fillna(df_realtime['pct_full'])
    df_realtime['pct_full_log2'] = df_realtime['pct_full_log2'].fillna(df_realtime['pct_full_log1'])
    df_realtime['pct_full_log3'] = df_realtime['pct_full_log3'].fillna(df_realtime['pct_full_log2'])
    df_realtime['pct_full_log_1h'] = df_realtime['pct_full_log_1h'].fillna(df_realtime['pct_full_log3'])
    df_realtime['rolling_mean_30m'] = df_realtime['rolling_mean_30m'].fillna(df_realtime['pct_full'])
    df_realtime['rolling_std_30m'] = df_realtime['rolling_std_30m'].fillna(0.0)
    df_realtime['rolling_mean_1h'] = df_realtime['rolling_mean_1h'].fillna(df_realtime['pct_full'])

    # Fórmulas físicas
    df_realtime['velocidad_cambio'] = df_realtime['pct_full'] - df_realtime['pct_full_log1']
    df_realtime['aceleracion_cambio'] = df_realtime['pct_full'] - (2 * df_realtime['pct_full_log1']) + df_realtime['pct_full_log2']
    df_realtime['tendencia_1h'] = df_realtime['pct_full'] - df_realtime['pct_full_log_1h']
    df_realtime['desviacion_de_media_30m'] = df_realtime['pct_full'] - df_realtime['rolling_mean_30m']
    df_realtime['momentum_quiebre_30m'] = df_realtime['velocidad_cambio'] - df_realtime['rolling_std_30m']

    df_realtime[FEATURES_REGRESOR] = df_realtime[FEATURES_REGRESOR].fillna(0)
    
    # Fase A: El Regresor predice el número exacto de bicicletas flotantes que habrá en +45 minutos.
    df_realtime['pred_num_regresor'] = np.clip(
        np.round(model_regresor.predict(df_realtime[FEATURES_REGRESOR])),
        0, df_realtime['capacity']
    )

    # Fase B: El Clasificador toma todas las variables previas + la predicción numérica recién hecha
    features_clasificador = FEATURES_REGRESOR + ['pred_num_regresor']
    df_realtime['estado_predicho'] = model_clasificador.predict(df_realtime[features_clasificador])
    df_realtime['bicis_predichas_45m'] = df_realtime['pred_num_regresor']

    df_realtime['inventario_optimo'] = np.round(df_realtime['capacity'] * 0.55)

    def calcular_unidades_netas(row):
        """
        Determina cuántas bicicletas faltan o sobran con base en el semáforo futuro predicho.
        Si la estación estará en rango normal (1 o 2), no requiere intervención (retorna 0).
        """
        if row['estado_predicho'] == 0:
            return int(max(1, row['inventario_optimo'] - row['bicis_predichas_45m']))
        elif row['estado_predicho'] == 3:
            return int(-max(1, row['bicis_predichas_45m'] - row['inventario_optimo']))
        return 0

    df_realtime['unidades_requeridas'] = df_realtime.apply(calcular_unidades_netas, axis=1)

    # 6. MATRIZ LOGÍSTICA DE REBALANCEO ENTRE ESTACIONES
    def calcular_distancia_directa(lat1, lon1, lat2, lon2):
        return abs(lat1 - lat2) + abs(lon1 - lon2)

    def asignar_vehiculo(amount):
        """
        Regla operativa para asignación de vehiculo
        """
        if amount > 32: return "Camión Grande (Cap 50)"
        elif amount > 16: return "Camioneta + Remolque (Cap 32)"
        else: return "Camioneta Ligera (Cap 16 - Sin Remolque)"

    # Filtrar solo aquellas estaciones que los modelos determinaron en situación crítica
    df_despacho_instante = df_realtime[df_realtime['unidades_requeridas'] != 0].copy()
    rutas_generadas = []

    for zona, grupo in df_despacho_instante.groupby('zona_logistica'):
        estaciones_oferta = grupo[grupo['unidades_requeridas'] < 0].copy()
        estaciones_demanda = grupo[grupo['unidades_requeridas'] > 0].copy()

        oferta_list = estaciones_oferta[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')
        demanda_list = estaciones_demanda[['station_id', 'lat', 'lon', 'unidades_requeridas']].to_dict('records')

        for o in oferta_list: o['disponibles'] = abs(o['unidades_requeridas'])
        for d in demanda_list: d['necesitadas'] = d['unidades_requeridas']

        for d in demanda_list:
            # Buscar estaciones vecinas de la misma zona con excedente disponible
            while d['necesitadas'] > 0:
                ofertas_validas = [o for o in oferta_list if int(o['station_id']) != int(d['station_id']) and o['disponibles'] > 0]
                if not ofertas_validas: break

                # Calcular distancias desde el nodo de demanda actual hacia todas las ofertas
                for o in ofertas_validas:
                    o['distancia'] = calcular_distancia_directa(d['lat'], d['lon'], o['lat'], o['lon'])

                # Validar restricción del radio de cobertura (~1km máximo)
                ofertas_dentro_del_radio = [o for o in ofertas_validas if o['distancia'] <= DISTANCIA_LIMITE_GRADOS]
                if not ofertas_dentro_del_radio: break

                # Ordenar para tomar siempre la estación de origen más cercana
                ofertas_dentro_del_radio = sorted(ofertas_dentro_del_radio, key=lambda x: x['distancia'])
                mas_cercana = ofertas_dentro_del_radio[0]
                cantidad_a_transferir = min(d['necesitadas'], mas_cercana['disponibles'])

                if cantidad_a_transferir > 0:
                    distancia_estimada_km = mas_cercana['distancia'] * 111.0 # Factor de conversión de grados a Km
                    rutas_generadas.append({
                        'zonaLogistica': int(zona), 'estacionOrigen': int(mas_cercana['station_id']),
                        'estacionDestino': int(d['station_id']), 'bicicletasAMover': int(cantidad_a_transferir),
                        'distanciaKm': round(distancia_estimada_km, 2), 'vehiculoAsignado': asignar_vehiculo(cantidad_a_transferir)
                    })
                    d['necesitadas'] -= cantidad_a_transferir
                    for o in oferta_list:
                        if o['station_id'] == mas_cercana['station_id']: o['disponibles'] -= cantidad_a_transferir

        # Si quedó demanda insatisfecha que no tenía vecinos cerca, se surte desde el Almacén Central
        for d in demanda_list:
            if d['necesitadas'] > 0:
                rutas_generadas.append({
                    'zonaLogistica': int(zona), 'estacionOrigen': "Almacén Central",
                    'estacionDestino': int(d['station_id']),
                    'bicicletasAMover': int(d['necesitadas']), 'distanciaKm': None,
                    'vehiculoAsignado': asignar_vehiculo(d['necesitadas'])
                })
        # Si quedaron estaciones saturadas sin estaciones con espio vecinas, se drena el exceso hacia el Almacén Central
        for o in oferta_list:
            if o['disponibles'] > 0:
                rutas_generadas.append({
                    'zonaLogistica': int(zona), 'estacionOrigen': int(o['station_id']),
                    'estacionDestino': "Almacén Central",
                    'bicicletasAMover': int(o['disponibles']), 'distanciaKm': None,
                    'vehiculoAsignado': asignar_vehiculo(o['disponibles'])
                })

    # 7. CONSOLIDACIÓN DE RESPUESTA FINAL (PAYLOAD API)
    df_rutas = pd.DataFrame(rutas_generadas)

    rutas_list = []
    if not df_rutas.empty:
        df_rutas['distanciaKm'] = df_rutas['distanciaKm'].replace({np.nan: None})
        rutas_list = df_rutas.to_dict(orient='records')

    df_criticas = df_realtime[df_realtime['estado_predicho'].isin([0, 3])].copy()
    total_compensado, total_necesitado = 0, 0
    for zona, grupo in df_criticas.groupby('zona_logistica'):
        demand_llenar = grupo[grupo['unidades_requeridas'] > 0]['unidades_requeridas'].sum()
        total_compensado += min(demand_llenar, abs(grupo[grupo['unidades_requeridas'] < 0]['unidades_requeridas'].sum()))
        total_necesitado += demand_llenar

    distancia_local = df_rutas['distanciaKm'].dropna().sum() if not df_rutas.empty else 0.0
    flota_resumen = df_rutas['vehiculoAsignado'].value_counts().to_dict() if not df_rutas.empty else {}

    metricas = {
        "green_logistics": {
            "movimientos_mitigados_unidades": int(total_compensado),
            "eficiencia_rebalanceo_local_pct": round((total_compensado / total_necesitado * 100), 2) if total_necesitado > 0 else 0.0,
            "distancia_total_optimizada_local_km": round(distancia_local, 2)
        },
        "flota_resumen": flota_resumen
    }

    return {
        "status": "success",
        "timestamp_evaluacion": ts_reciente.strftime('%Y-%m-%d %H:%M:%S.%f'),
        "metricas_globales": metricas,
        "hoja_de_ruta": rutas_list
    }