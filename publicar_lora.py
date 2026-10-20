# Path nos permite trabajar con carpetas y archivos
from pathlib import Path
# HfApi permite crear repositorios y subir archivos al Hub
from huggingface_hub import HfApi

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpeta donde guardamos nuestro adaptador entrenado
ADAPTER_PATH = Path("modelo_lora/adapter_final")

# Identificador del repositorio que crearemos en Hugging Face
REPO_ID = "JulioLopzz/qwen3-0.6b-soporte-ti-lora"

# ============================================================
# PUBLICAR ADAPTADOR LORA
# ============================================================

def publicar_modelo():
    """Crea el repositorio y publica nuestro adaptador LoRA"""

    #Comprobamos que existe la carpeta del adaptador
    assert ADAPTER_PATH.exists(), "No se encuentra el adaptador LoRA"

    #Creamos el cliente para comunicarnos con Hugging Face
    api = HfApi()

    #Creamos un repositorio publico para nuestro modelo
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        private=False,
        exist_ok=True
    )

    #Subimos los archivos del adaptador al repositorio
    api.upload_folder(
        folder_path=str(ADAPTER_PATH),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Publicar adaptador LoRA de soporta TI"
    )

    #Mostramos la direccion del modelo publicado
    print(f"\nModelo publicado en: https://huggingface.co/{REPO_ID}")

# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("         PUBLICACIÓN DE QWEN3 + LORA")
    print("=" * 55)

    # Publicamos el adaptador en Hugging Face Hub
    publicar_modelo()

    # Confirmamos que el proceso ha terminado
    print("\n¡Publicación completada correctamente!")
    print("=" * 55 + "\n")