# Path nos permite trabajar con rutas de archivos y carpetas
from pathlib import Path
# PyTorch nos permite detectar si tenemos una GPU disponible
import torch
#Permite guardar los resultados y las metricas en formato json
import json
# Cargamos el modelo base y su tokenizador
from transformers import AutoTokenizer, AutoModelForCausalLM
# PeftModel permite incorporar nuestro adaptador LoRA
from peft import PeftModel
# Reutilizamos las funciones de evaluación del proyecto
from evaluar_modelo import cargar_jsonl, generar_respuesta, evaluar

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Modelo original que utilizamos durante el entrenamiento
MODEL_ID = "Qwen/Qwen3-0.6B"

# Carpeta donde guardamos el adaptador entrenado
ADAPTER_PATH = Path("modelo_lora/adapter_final")

# Dataset de test, que debe permanecer sin modificaciones
TEST_PATH = Path("data/test.jsonl")

# Carpeta donde guardaremos los resultados de la evaluación
RESULTADOS_DIR = Path("resultados")

# Seleccionamos la GPU si está disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CARGAR MODELO CON ADAPTADOR LORA
# ============================================================

def cargar_modelo_lora():

    # Cargamos el tokenizador utilizado durante el entrenamiento
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    # Cargamos el modelo base Qwen3-0.6B
    modelo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    )

    # Incorporamos al modelo base los pesos de nuestro LoRA
    modelo = PeftModel.from_pretrained(
        modelo_base,
        ADAPTER_PATH
    )

    # Movemos el modelo a la GPU si está disponible
    modelo = modelo.to(DEVICE)

    # Activamos el modo evaluación (sin entrenamiento)
    modelo.eval()

    # Devolvemos el modelo especializado y su tokenizador
    return modelo, tokenizer


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("              EVALUACIÓN QWEN3 + LORA")
    print("=" * 55)

    # --------------------------------------------------------
    # 1. CARGAR EL MODELO
    # --------------------------------------------------------
    print("\n[1/3] CARGANDO MODELO LORA")

    #Comprobamos que el adaptador existe
    assert ADAPTER_PATH.exists(), "No se encuentra el adaptador LoRA"

    #Cargamos Qwen3 junto con nuestro adaptador entrenado
    modelo, tokenizer = cargar_modelo_lora()

    print("  Modelo cargado correctamente.")
    print(f"  Dispositivo: {DEVICE}")

    # --------------------------------------------------------
    # 2. CARGAR LOS DATOS DE TEST
    # --------------------------------------------------------
    print("\n[2/3] CARGANDO DATASET DE TEST")

    #Recuperamos las mismas incidencias utilizadas anteriormente
    test = cargar_jsonl(TEST_PATH)

    print(f"  Incidencias: {len(test)}")

    #Comprobamos que seguimos utilizando las 30 originales
    assert len(test) == 30, "El test debe contener 30 incidencias"

    # --------------------------------------------------------
    # 3. EVALUAR EL MODELO
    # --------------------------------------------------------

    print("\n[3/3] EVALUANDO MODELO LORA")
    print("-" * 55)

    # Evaluamos las 30 incidencias con nuestro modelo LoRA
    resultados, metricas = evaluar(test, modelo, tokenizer)

    # Creamos la carpeta de resultados si todavía no existe
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    # Definimos los archivos donde guardaremos la evaluación
    ruta_predicciones = RESULTADOS_DIR / "lora.jsonl"
    ruta_metricas = RESULTADOS_DIR / "lora_metricas.json"

    # Guardamos una predicción por línea en formato JSONL
    with open(ruta_predicciones, "w", encoding="utf-8") as archivo:
        for resultado in resultados:
            archivo.write(json.dumps(resultado, ensure_ascii=False) + "\n")

    # Guardamos las métricas en un archivo JSON
    with open(ruta_metricas, "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)

    # Mostramos un resumen claro de los resultados
    print("\n" + "=" * 55)
    print("             RESULTADOS EVALUACIÓN LORA")
    print("=" * 55)

    print(f"  JSON válido:  {metricas['json_valido_pct']} %")
    print(f"  Categoría:    {metricas['categria_accuracy_pct']} %")
    print(f"  Prioridad:    {metricas['prioridades_accuracy_pct']} %")

    print("\n  Resultados guardados correctamente.")
    print("=" * 55 + "\n")




