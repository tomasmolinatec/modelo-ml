from fastapi import FastAPI, HTTPException
from prediccion_realtime import cargar_modelos, ejecutar_prediccion

app = FastAPI(title="Ecobici ML Service")

_modelos = None


def get_modelos():
    global _modelos
    if _modelos is None:
        _modelos = cargar_modelos()
    return _modelos


@app.on_event("startup")
def on_startup():
    get_modelos()
    print("[+] Modelos cargados. Servicio listo.")


@app.post("/predecir")
def predecir():
    try:
        resultado = ejecutar_prediccion(*get_modelos())
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al ejecutar predicción: {e}")


@app.get("/")
def root():
    return {"status": "ok"}
