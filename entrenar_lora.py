# Para trabajar con archivos y respuestas en formato JSON
import json
from pathlib import Path
# PyTorch: operaciones y entrenamiento en GPU
import torch
# Datasets: cargar y preparar los datos de entrenamiento
from datasets import load_dataset
# Transformers: cargar el modelo y su tokenizador
from transformers import AutoTokenizer, AutoModelForCausalLM
# PEFT: configuracion y aplicacion del adaptador LoRA
from peft import LoraConfig, get_peft_model
# TRL: herramientas para el fine-tuning supervisado
from trl import SFTTrainer, SFTConfig # type: ignore
# Reutilizamos las instrucciones utilizadas en Zero-shot y Few-shot
from evaluar_modelo import INSTRUCCIONES

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

#Modelo de HuggingFace que vamos a especializar
MODEL_ID = "Qwen/Qwen3-0.6B"
#Utilizamos la GPU si esta disponible, sino, la CPU
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
# Archivos con los datos de entrenamiento y validación
TRAIN_PATH = Path("data/train.jsonl")
VALIDATION_PATH = Path("data/validation.jsonl")
# Carpeta donde guardaremos nuestro adaptador LoRA
OUTPUT_DIR = Path("modelo_lora")
# Creamos la carpeta si todavía no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Mostramos el dispositivo que utilizaremos
print(f"Dispositivo disponible: {DEVICE}")

# ============================================================
# 2. PREPARACIÓN DE LOS DATOS
# ============================================================

def preparar_ejemplo(ejemplo):
    """Convierte una incidencia en una conversación para Qwen"""
    #Construimos la respuesta correcta a partir de las etiquetas
    respuesta = {
        "categoria": ejemplo["categoria"],
        "prioridad": ejemplo["prioridad"],
        "respuesta": ejemplo["respuesta"]
    }

    #Creamos la conversacion con sus tres roles
    mensajes = [
        {
            "role": "system",
            "content": INSTRUCCIONES
        },
        {
            "role": "user",
            "content": ejemplo["incidencia"]
        },
        {
            "role": "assistant",
            "content": json.dumps(respuesta, ensure_ascii=False)
        }
    ]

    #Devolvemos la conversacion en una columna llamada mensajes
    return{"messages": mensajes}

def cargar_datos():
    """Carga los datasets y los transforma en conversaciones"""

    #Cargamos los archivos JSON de entrenamiento y validación
    datasets = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_PATH),
            "validation": str(VALIDATION_PATH)
        }
    )

    #Convertimos cada incidencia en una conversacion
    datasets = datasets.map(
        preparar_ejemplo,
        remove_columns=datasets["train"].column_names
    )

    # Devolvemos los dos conjuntos preparados
    return datasets["train"], datasets["validation"]

# ============================================================
# 3. CONFIGURACIÓN DEL ADAPTADOR LORA
# ============================================================

def configurar_lora():
    """Define los parametros del adaptador LoRA"""

    # Creamos la configuración de LoRA
    lora_config = LoraConfig(
        r=8,                        # Rango de las matrices entrenables
        lora_alpha=16,              # Escalado de la adaptación
        lora_dropout=0.05,          # Regularización para reducir sobreajuste
        bias="none",                # No entrenamos los parámetros de sesgo
        task_type="CAUSAL_LM",      # Qwen genera texto de forma autorregresiva
        target_modules=[            # Capas donde aplicaremos LoRA
            "q_proj",
            "v_proj"
        ]
    )

    return lora_config  #Devolvemos la configuracion

# ============================================================
# 4. CARGA DEL MODELO Y APLICACIÓN DE LORA
# ============================================================

def cargar_modelo():
    """Cargar Qwen y añadir los parametros entrenables de LoRA"""

    #Cargamos el tokenizador correspondiente a Qwen
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    #Cargams el modelo original utilizando precision FP16
    modelo = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    )

    #Movemos el modelo a la GPU si está disponible
    modelo = modelo.to(DEVICE) # type: ignore

    #Creamos nuestra configuracion LoRA
    lora_config = configurar_lora()

    #Aplicamos LoRA sobre las capas seleccionadas
    modelo = get_peft_model(modelo, lora_config)

    #Mostramos cuantos parametros se entrenan realmente
    modelo.print_trainable_parameters()

    #Devolvemos el modelo adaptado y su tokenizador
    return modelo, tokenizer


# ============================================================
# 5. CONFIGURACIÓN DEL ENTRENAMIENTO
# ============================================================

def configurar_entrenamiento():
    """Define cómo entrenaremos el adaptador LoRA."""

    config = SFTConfig(
        output_dir=str(OUTPUT_DIR),       # Carpeta para guardar resultados

        num_train_epochs=3,               # Recorremos el dataset 3 veces
        per_device_train_batch_size=2,    # Procesamos 2 ejemplos por lote
        per_device_eval_batch_size=2,     # Validamos con lotes de 2 ejemplos

        gradient_accumulation_steps=4,    # Acumulamos 4 lotes antes de actualizar
        learning_rate=2e-4,               # Tasa de aprendizaje: 0.0002

        fp16=(DEVICE == "cuda"),          # Entrenamiento en FP16 si hay GPU
        max_length=512,                   # Longitud máxima de cada conversación

        assistant_only_loss=True,         # Aprendemos solo de la respuesta del asistente

        eval_strategy="epoch",            # Validamos al terminar cada época
        save_strategy="epoch",            # Guardamos un checkpoint por época

        logging_steps=5,                  # Mostramos métricas cada 5 pasos
        report_to="none"                  # Sin integraciones externas de seguimiento
    )

    return config

# ============================================================
# 6. CREACIÓN DEL ENTRENADOR
# ============================================================

def crear_entrenador(modelo, tokenizer, train, validation):
    """Conecta el modelo, los datos y la configuracion de SFT"""

    #Recuperamos los hiperparámetros del entrenamiento
    config = configurar_entrenamiento()

    #Creamos el entrenador supervisado
    entrenador = SFTTrainer(
        model=modelo,                  # Qwen con LoRA aplicado
        args=config,                   # Épocas, batch size, learning rate...
        train_dataset=train,           # Nuestros 200 ejemplos
        eval_dataset=validation,       # Nuestros 30 ejemplos de validación
        processing_class=tokenizer     # Tokenizador de Qwen
    )

    #Devolvemos el entrenador preparado
    return entrenador

# ============================================================
# INICIAR ENTRENAMIENTO Y MOSTRAR RESULTADOS
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("             FINE-TUNING QWEN3 CON LORA")
    print("=" * 55)

    # --------------------------------------------------------
    # 1. PREPARACIÓN DE LOS DATOS
    # --------------------------------------------------------

    print("\n[1/5] PREPARANDO DATASETS")

    # Cargamos los datos de entrenamiento y validación
    train, validation = cargar_datos()

    print(f"  Entrenamiento: {len(train)} ejemplos")
    print(f"  Validación:    {len(validation)} ejemplos")

    # --------------------------------------------------------
    # 2. CARGA DEL MODELO
    # --------------------------------------------------------

    print("\n[2/5] CARGANDO QWEN3 + LORA")

    # Cargamos Qwen con los adaptadores LoRA incorporados
    modelo, tokenizer = cargar_modelo()

    print("  Modelo cargado correctamente.")
    print(f"  Dispositivo: {DEVICE}")

    # --------------------------------------------------------
    # 3. CONFIGURACIÓN DEL ENTRENAMIENTO
    # --------------------------------------------------------

    print("\n[3/5] CONFIGURACIÓN DEL ENTRENAMIENTO")

    # Recuperamos los hiperparámetros del entrenamiento
    config = configurar_entrenamiento()

    print(f"  Épocas:             {config.num_train_epochs}")
    print(f"  Batch size:         {config.per_device_train_batch_size}")
    print(f"  Acumulación:        {config.gradient_accumulation_steps}")
    print(f"  Learning rate:      {config.learning_rate}")
    print(f"  Assistant-only:     {config.assistant_only_loss}")

    # --------------------------------------------------------
    # 4. CREACIÓN Y VERIFICACIÓN DEL ENTRENADOR
    # --------------------------------------------------------

    print("\n[4/5] PREPARANDO SFTTRAINER")

    # Conectamos el modelo, los datos y la configuración
    entrenador = crear_entrenador(
        modelo,
        tokenizer,
        train,
        validation
    )

    # Comprobamos que el dataset contiene ejemplos entrenables
    assert entrenador.train_dataset is not None

    # Recuperamos el primer ejemplo procesado
    ejemplo = entrenador.train_dataset[0]

    # Contamos los tokens que participan en el entrenamiento
    tokens_entrenables = sum(
        label != -100
        for label in ejemplo["labels"]
    )

    # Evitamos entrenar si no hay tokens válidos
    assert tokens_entrenables > 0, "No hay tokens entrenables."

    print("  SFTTrainer preparado correctamente.")
    print(f"  Tokens del ejemplo:      {len(ejemplo['input_ids'])}")
    print(f"  Tokens entrenables:      {tokens_entrenables}")
    print("  Máscara de entrenamiento: OK")

    # --------------------------------------------------------
    # 5. ENTRENAMIENTO Y GUARDADO
    # --------------------------------------------------------

    print("\n[5/5] INICIANDO ENTRENAMIENTO LORA")
    print("-" * 55)

    # Entrenamos los parámetros LoRA durante las épocas definidas
    entrenador.train()

    # Definimos una única carpeta para el resultado final
    ruta_final = OUTPUT_DIR / "adapter_final"

    # Guardamos el adaptador LoRA entrenado
    entrenador.save_model(str(ruta_final))

    # Guardamos el tokenizador en la misma carpeta
    tokenizer.save_pretrained(str(ruta_final))

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("           ENTRENAMIENTO COMPLETADO")
    print("=" * 55)

    print(f"  Modelo base:  {MODEL_ID}")
    print(f"  Épocas:       {config.num_train_epochs}")
    print(f"  Adaptador:    {ruta_final}")
    print("\n  ¡LoRA entrenado y guardado correctamente!")
    print("=" * 55 + "\n")

