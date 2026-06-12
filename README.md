# Ecobici ML Service

Servicio de machine learning en tiempo real para predecir la disponibilidad de bicicletas y generar rutas de rebalanceo en la red de Ecobici de la Ciudad de México.

Corre con :
uv run uvicorn main:app --reload

En el servidor:
uv run uvicorn main:app --host 0.0.0.0 --port 8000

## Descripción General

El servicio consume datos en vivo de las estaciones desde el feed GBFS de Ecobici, ejecuta un pipeline de predicción XGBoost en dos etapas y devuelve un plan logístico accionable — indicando qué estaciones estarán críticamente vacías o llenas en los próximos 45 minutos y cómo redistribuir las bicicletas entre ellas.

## Funcionamiento

### Pipeline de Predicción

La lógica principal se ejecuta en dos fases secuenciales:

1. **Regresor** — predice el número exacto de bicicletas disponibles en cada estación dentro de 45 minutos.
2. **Clasificador** — toma la salida del regresor como variable adicional y clasifica cada estación en uno de cuatro estados de ocupación:

| Estado | Significado |
|--------|-------------|
| `0` | Déficit crítico (≤ 30% de capacidad) |
| `1` | Normal — bajo (30–55%) |
| `2` | Normal — alto (55–85%) |
| `3` | Saturación crítica (> 85%) |

Solo las estaciones predichas en estado `0` o `3` se marcan para intervención.

### Memoria Temporal (Buffer de Lags)

Dado que la API de Ecobici solo expone el estado actual, el servicio mantiene un buffer CSV rotativo (`ml/buffer_tiempo_real.csv`) con los últimos 65 minutos de snapshots. Esto proporciona las variables de rezago que los modelos necesitan para detectar la inercia — si una estación se está vaciando o llenando.

### Matriz Logística

Para cada una de las 5 zonas geográficas (clusters K-Means), el servicio empareja estaciones con exceso de bicicletas (estado `3`) con estaciones cercanas que las necesitan (estado `0`) dentro de un radio de ~1 km, minimizando la distancia total recorrida. La demanda o el excedente no resueltos se enrutan hacia o desde el Almacén Central.

## Estructura del Proyecto

```
.
├── main.py                     # Punto de entrada de la aplicación FastAPI
├── prediccion_realtime.py      # Pipeline ML y motor logístico
├── eficiencia-ecobici.ipynb    # Notebook de entrenamiento de modelos
└── ml/
    ├── regresor_ecobici.json           # Regresor XGBoost entrenado
    ├── clasificador_ecobici.json       # Clasificador XGBoost entrenado
    ├── mapa_tri_interaccion.csv        # Mapa histórico estación × hora × día de semana
    ├── mapa_historico.csv              # Promedios de ocupación por estación y hora
    ├── encoding_map.csv                # Target encoding por estación y tipo de horario
    ├── coords_base.csv                 # Coordenadas de estaciones con asignación de zona
    └── buffer_tiempo_real.csv          # Buffer de lags rotativo (generado en tiempo de ejecución)
```

## API

### `POST /predecir`

Obtiene datos en vivo de las estaciones, ejecuta el pipeline completo de predicción y logística, y devuelve los resultados.

**Respuesta:**
```json
{
  "status": "success",
  "timestamp_evaluacion": "2024-11-15 08:32:11.123456",
  "metricas_globales": {
    "green_logistics": {
      "movimientos_mitigados_unidades": 142,
      "eficiencia_rebalanceo_local_pct": 78.5,
      "distancia_total_optimizada_local_km": 34.2
    },
    "flota_resumen": {
      "Camioneta Ligera (Cap 16 - Sin Remolque)": 8,
      "Camioneta + Remolque (Cap 32)": 3
    }
  },
  "hoja_de_ruta": [
    {
      "zonaLogistica": 2,
      "estacionOrigen": 314,
      "estacionDestino": 287,
      "bicicletasAMover": 11,
      "distanciaKm": 0.73,
      "vehiculoAsignado": "Camioneta Ligera (Cap 16 - Sin Remolque)"
    }
  ]
}
```

### `GET /`

Health check. Devuelve `{"status": "ok"}`.

## Instalación y Uso

### Dependencias

```
fastapi
uvicorn
xgboost
pandas
numpy
requests
```

Instalar con:
```bash
pip install fastapi uvicorn xgboost pandas numpy requests
```

### Iniciar el Servicio

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Los modelos se cargan una sola vez al arrancar. El directorio `ml/` debe estar presente y contener todos los archivos de modelos y mapas antes de iniciar.

### Entrenar los Modelos

Abrir y ejecutar `eficiencia-ecobici.ipynb` con un dataset histórico (`ecobici_datos_20k.csv`). El notebook entrena ambos modelos XGBoost y exporta todos los artefactos necesarios al directorio `ml/`.

## Reglas de Asignación de Vehículos

| Bicicletas a mover | Vehículo |
|--------------------|----------|
| ≤ 16 | Camioneta Ligera (Cap 16) |
| 17 – 32 | Camioneta + Remolque (Cap 32) |
| > 32 | Camión Grande (Cap 50) |