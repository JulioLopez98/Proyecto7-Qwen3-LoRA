# ============================================================
# PROBAR EL ADAPTADOR LORA PUBLICADO EN HUGGING FACE
# ============================================================

# PyTorch nos permite utilizar la GPU
import torch

# Cargamos el modelo base y su tokenizador
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT permite cargar nuestro adaptador LoRA publicado
from peft import PeftModel

# Reutilizamos la función de generación del proyecto
from evaluar_modelo import generar_respuesta

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Modelo base original
MODEL_ID = "Qwen/Qwen3-0.6B"

# Repositorio público donde hemos publicado el adaptador
REPO_ID = "JulioLopzz/qwen3-0.6b-soporte-ti-lora"

# Utilizamos la GPU si está disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("           PRUEBA DEL MODELO DESDE HF HUB")
    print("=" * 55)

    # Descargamos el tokenizador desde nuestro repositorio
    print("\n[1/3] CARGANDO TOKENIZADOR DESDE HUGGING FACE")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    # Cargamos el modelo Qwen3 original
    print("\n[2/3] CARGANDO QWEN3 + ADAPTADOR REMOTO")
    modelo_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    )

    # Descargamos y aplicamos el adaptador LoRA desde el Hub
    modelo = PeftModel.from_pretrained(
        modelo_base,
        REPO_ID
    )

    # Preparamos el modelo para realizar predicciones
    modelo = modelo.to(DEVICE)
    modelo.eval()

    # Definimos una incidencia nueva para probar el modelo
    incidencia = "No puedo conectarme a la red wifi de la oficina."

    # Generamos una clasificación con el modelo descargado
    print("\n[3/3] CLASIFICANDO INCIDENCIA")

    respuesta = generar_respuesta(
        modelo,
        tokenizer,
        incidencia
    )

    # Mostramos el resultado
    print(f"\nIncidencia: {incidencia}")
    print(f"Respuesta:  {respuesta}")

    print("\n" + "=" * 55)
    print("                PRUEBA FINALIZADA")
    print("=" * 55)