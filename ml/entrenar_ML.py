import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

# 1. CARGA Y PREPARACIÓN DE DATOS
df = pd.read_csv('ecobici_datos_20k.xls')

# Homogeneización temporal y ordenamiento cronológico por estación
df['run_ts'] = pd.to_datetime(df['run_ts'], errors='coerce')
df = df.dropna(subset=['run_ts'])
df = df.sort_values(by=['station_id', 'run_ts']).reset_index(drop=True)

# HORIZONTE LOGÍSTICO (50 minutos objetivos)
# Se genera el target a predecir desplazando el estado de la estación 5 pasos hacia el futuro
df['bikes_avail_future_45m'] = df.groupby('station_id')['bikes_avail'].shift(-5)
df['pct_full_future_45m'] = df.groupby('station_id')['pct_full'].shift(-5)

def asignar_rango_ocupacion_estricto(pct):
    if pd.isna(pct): return np.nan
    if pct <= 0.30: return 0     # Déficit
    elif pct <= 0.55: return 1   # Normal-Bajo
    elif pct <= 0.85: return 2   # Normal-Alto
    else: return 3               # Saturación               

df['target_rango_45m'] = df['pct_full_future_45m'].apply(asignar_rango_ocupacion_estricto)

# 2. INGENIERÍA DE VARIABLES (MOMENTUM, INERCIA Y CONTROL)
df['hora'] = df['run_ts'].dt.hour
df['dia_semana'] = df['run_ts'].dt.dayofweek
df['es_fin_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)

# Transformación cíclica de la hora
df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24.0)
df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24.0)
df['hora_tipo_dia'] = df['hora'] + (df['es_fin_de_semana'] * 24)
df['es_pico_semana'] = df['dia_semana'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# Históricos y Rezagos de Inercia (Refactorizado: Sin log_2h)
df['pct_full_log1'] = df.groupby('station_id')['pct_full'].shift(1)
df['pct_full_log2'] = df.groupby('station_id')['pct_full'].shift(2)
df['pct_full_log3'] = df.groupby('station_id')['pct_full'].shift(3)
df['pct_full_log_1h'] = df.groupby('station_id')['pct_full'].shift(5)

# Ventanas móviles basadas en el pasado inmediato previo
df['rolling_mean_30m'] = df.groupby('station_id')['pct_full'].transform(lambda x: x.shift(1).rolling(3).mean())
df['rolling_std_30m'] = df.groupby('station_id')['pct_full'].transform(lambda x: x.shift(1).rolling(3).std()).fillna(0)
df['rolling_mean_1h'] = df.groupby('station_id')['pct_full'].transform(lambda x: x.shift(1).rolling(6).mean())

# Flujos de Diferenciación y Momentum Corto
df['velocidad_cambio'] = df['pct_full_log1'] - df['pct_full_log2']
df['aceleracion_cambio'] = df['pct_full_log1'] - (2 * df['pct_full_log2']) + df['pct_full_log3']
df['tendencia_1h'] = df['pct_full'] - df['pct_full_log_1h']
df['desviacion_de_media_30m'] = df['pct_full'] - df['rolling_mean_30m'] 
df['momentum_quiebre_30m'] = df['velocidad_cambio'] - df['rolling_std_30m']

# Variables de Capacidad Física y Cruces
df['slots_vacios'] = df['capacity'] - df['bikes_avail']
df['interaccion_cap_hora'] = df['capacity'] * df['hora_cos']

# Target Encoding de Alta Dimensión Cruzado
mapa_tri_interaccion = df.groupby(['station_id', 'hora', 'dia_semana'])['pct_full'].mean().reset_index()
mapa_tri_interaccion.rename(columns={'pct_full': 'historico_estacion_hora_dia'}, inplace=True)
df = df.merge(mapa_tri_interaccion, on=['station_id', 'hora', 'dia_semana'], how='left')

mapa_historico = df.groupby(['station_id', 'hora'])['pct_full'].mean().reset_index()
mapa_historico.rename(columns={'pct_full': 'ocupacion_historica_estacion_hora'}, inplace=True)
df = df.merge(mapa_historico, on=['station_id', 'hora'], how='left')

encoding_map = df.groupby(['station_id', 'hora_tipo_dia'])['target_rango_45m'].mean().reset_index()
encoding_map.rename(columns={'target_rango_45m': 'rango_historico_estacion'}, inplace=True)
df = df.merge(encoding_map, on=['station_id', 'hora_tipo_dia'], how='left')

# Segmentación por regiones logísticas K-Means
if 'zona_logistica' in df.columns: df = df.drop(columns=['zona_logistica'])
coords = df[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
kmeans_final = KMeans(n_clusters=6, random_state=42, n_init=10)
coords['zona_logistica'] = kmeans_final.fit_predict(coords[['lat', 'lon']])
df = df.merge(coords, on=['lat', 'lon'], how='left')

# GUARDAR MAPAS BASE PARA PRODUCCIÓN
df[['station_id', 'hora', 'dia_semana', 'historico_estacion_hora_dia']].drop_duplicates().to_csv("mapa_tri_interaccion.csv", index=False)
df[['station_id', 'hora', 'ocupacion_historica_estacion_hora']].drop_duplicates().to_csv("mapa_historico.csv", index=False)
df[['station_id', 'hora_tipo_dia', 'rango_historico_estacion']].drop_duplicates().to_csv("encoding_map.csv", index=False)
coords.to_csv("coords_base.csv", index=False)

# 3. LIMPIEZA Y DIVISIÓN (80% Train, 20% Test)
columnas_limpieza = ['target_rango_45m', 'bikes_avail_future_45m', 'pct_full_log1', 'rolling_mean_30m', 'historico_estacion_hora_dia', 'rango_historico_estacion', 'momentum_quiebre_30m']
df_xgb = df.dropna(subset=columnas_limpieza)
df_xgb = df_xgb[(df_xgb['is_installed'] == 1) & (df_xgb['is_renting'] == 1) & (df_xgb['is_returning'] == 1)]
df_xgb = df_xgb.sort_values('run_ts')

split_idx = int(len(df_xgb) * 0.8)
train_df = df_xgb.iloc[:split_idx].copy()
test_df = df_xgb.iloc[split_idx:].copy()

# FEATURES REGINAL (Sin pct_full_log_2h)
FEATURES_REGRESOR = [
    'capacity', 'bikes_avail', 'pct_full', 'hora_sin', 'hora_cos', 'dia_semana', 
    'es_fin_de_semana', 'hora_tipo_dia', 'es_pico_semana', 'pct_full_log1', 'pct_full_log2', 'pct_full_log3',
    'pct_full_log_1h', 'rango_historico_estacion', 'velocidad_cambio', 
    'aceleracion_cambio', 'tendencia_1h', 'ocupacion_historica_estacion_hora', 
    'slots_vacios', 'zona_logistica', 'rolling_mean_30m', 'rolling_std_30m', 'rolling_mean_1h',
    'interaccion_cap_hora', 'historico_estacion_hora_dia', 'desviacion_de_media_30m', 'momentum_quiebre_30m'
]

train_df[FEATURES_REGRESOR] = train_df[FEATURES_REGRESOR].fillna(0)
test_df[FEATURES_REGRESOR] = test_df[FEATURES_REGRESOR].fillna(0)

split_val_idx = int(len(train_df) * 0.85)
train_sub_df = train_df.iloc[:split_val_idx].copy()
val_sub_df = train_df.iloc[split_val_idx:].copy()

# 4. ENTRENAMIENTO FIJO DEL REGRESOR OPTIMIZADO
model_regresor_final = xgb.XGBRegressor(
    n_estimators=531, learning_rate=0.01786, max_depth=6,
    subsample=0.8436, colsample_bytree=0.8498, reg_alpha=3.983, reg_lambda=4.814,
    objective='reg:absoluteerror', eval_metric='mae', early_stopping_rounds=45,
    random_state=42, n_jobs=-1
)

model_regresor_final.fit(
    train_sub_df[FEATURES_REGRESOR], train_sub_df['bikes_avail_future_45m'],
    eval_set=[(val_sub_df[FEATURES_REGRESOR], val_sub_df['bikes_avail_future_45m'])],
    verbose=False
)
model_regresor_final.save_model("regresor_ecobici.json")

train_df['pred_num_regresor'] = np.clip(np.round(model_regresor_final.predict(train_df[FEATURES_REGRESOR])), 0, train_df['capacity'])
test_df['pred_num_regresor'] = np.clip(np.round(model_regresor_final.predict(test_df[FEATURES_REGRESOR])), 0, test_df['capacity'])

FEATURES_CLASIFICADOR = FEATURES_REGRESOR + ['pred_num_regresor']

# 5. ENTRENAMIENTO DEL CLASIFICADOR SECUENCIAL BALANCEADO
y_train_cl = train_df['target_rango_45m'].astype(int)
pesos_clase = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_cl), y=y_train_cl)
pesos_suavizados = (pesos_clase + 1.0) / 2.0
pesos_muestras_train = y_train_cl.map(lambda x: pesos_suavizados[x]).values

model_clasificador_final = xgb.XGBClassifier(
    n_estimators=265, learning_rate=0.02144, max_depth=7,
    subsample=0.8436, colsample_bytree=0.8498, objective='multi:softprob',
    num_class=4, random_state=42, n_jobs=-1
)
model_clasificador_final.fit(train_df[FEATURES_CLASIFICADOR], y_train_cl, sample_weight=pesos_muestras_train)
model_clasificador_final.save_model("clasificador_ecobici.json")

# 6. REPORTES DE PERFORMANCE
final_mae = mean_absolute_error(test_df['bikes_avail_future_45m'], test_df['pred_num_regresor'])
final_r2 = r2_score(test_df['bikes_avail_future_45m'], test_df['pred_num_regresor'])
print(f"[+] MAE Final Consolidado: {final_mae:.4f} bicicletas")
print(f"[+] Capacidad Explicativa (R²): {final_r2:.4f}")