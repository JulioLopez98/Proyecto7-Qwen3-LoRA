# Librerías para trabajar con JSON y rutas de archivos
import json
from pathlib import Path

# PyTorch para ejecutar el modelo en la GPU
import torch

# Transformers para cargar Qwen y su tokenizador
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reutilizamos las funciones de evaluación que ya construimos
from evaluar_modelo import cargar_jsonl, evaluar, INSTRUCCIONES

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
# Modelo que vamos a evaluar (el mismo que en Zero-shot)
MODEL_ID = "Qwen/Qwen3-0.6B"

# Utilizamos la GPU si está disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas de los datasets
TRAIN_PATH = Path("data/train.jsonl")
VALIDATION_PATH = Path("data/validation.jsonl")
TEST_PATH = Path("data/test.jsonl")

# Carpeta donde guardaremos los resultados
RESULTADOS_DIR = Path("resultados")

# Creamos la carpeta si todavía no existe
RESULTADOS_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. SELECCIONAR EJEMPLOS FEW-SHOT
# ============================================================
def seleccionar_ejemplos(train):
    """Selecciona ejemplos de entrenamiento para el prompt Few-shot"""
    ejemplos = []  #Aqui guardaremos los ejemplos seleccionados
    categorias_usadas = set()  #Evitamos repetir categorias

    #Recorremos las incidencias del conjunto de entrenamiento
    for incidencia in train:
        categoria = incidencia["categoria"]

        #Seleccionamos una incidenciade cada categoria
        if categoria not in categorias_usadas:
            ejemplos.append(incidencia)
            categorias_usadas.add(categoria)

        #Terminamos cuando tenemos las cuatro categorias
        if len(ejemplos) == 4:
            break
    return ejemplos

# ============================================================
# 3. CONSTRUIR EL PROMPT FEW-SHOT
# ============================================================
def construir_mensajes(incidencia, ejemplos):
    """Construye una conversación con ejemplos resueltos."""

    # Utilizamos las mismas instrucciones que en Zero-shot
    mensajes = [
        {
            "role": "system",
            "content": INSTRUCCIONES
        }
    ]

    #Añadimos cada ejemplo como una conversacion resulta
    for ejemplo in ejemplos:

        #Primero mostramos la incidencia de ejemplo
        mensajes.append({
            "role": "user",
            "content": ejemplo["incidencia"]
        })

        # Después mostramos su respuesta correcta en formato JSON
        respuesta = {
            "categoria": ejemplo["categoria"],
            "prioridad": ejemplo["prioridad"],
            "respuesta": ejemplo["respuesta"]
        }

        #Despues mostrmos su respuesta correcta en formato JSON
        mensajes.append({
            "role": "assistant",
            "content": json.dumps(respuesta, ensure_ascii=False)
        })

    # Finalmente añadimos la incidencia que queremos clasificar
    mensajes.append({
        "role": "user",
        "content": incidencia
    })

    return mensajes

# ============================================================
# 4. GENERAR RESPUESTAS FEW-SHOT
# ============================================================
def generar_respuesta_few_shot(model, tokenizer, incidencia, ejemplos):
    """Genera una respuesta utilizando ejemplos Few-shots"""

    #Construidos la conversacion con los ejemplos resueltos
    mensajes = construir_mensajes(incidencia, ejemplos)

    # Convertimos la conversación en tokens para Qwen
    entradas = tokenizer.apply_chat_template(
        mensajes,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to(DEVICE)

    # Generamos una respuesta sin aleatoriedad
    with torch.no_grad():
        salidas = model.generate(
            **entradas,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    #Separamos los tokens nuevos de los tokens del prompt
    token_respuesta = salidas[0][entradas["input_ids"].shape[-1]:]

    #convertimos los tokens generados en texto
    respuesta = tokenizer.decode(
        token_respuesta,
        skip_special_tokens = True
    )

    return respuesta

# ============================================================
# 5. PREPARAR EL GENERADOR FEW-SHOT
# ============================================================
def crear_generador_few_shot(ejemplos):
    """Prepara una función de generacion con ejemplos fijos"""

    #Esta función utiliza los ejemplos seleccionados
    def generar(model, tokenizer, incidencia):

        #Llamamos al generador few_shot con los cuatro argumentos
        return generar_respuesta_few_shot(
            model,
            tokenizer,
            incidencia,
            ejemplos
        )
    # Devolvemos la función preparada, sin ejecutarla todavía
    return generar


# ============================================================
# 6. FUNCIÓN PRINCIPAL
# ============================================================
def main():
    """Carga el modelo y evalúa Few-shot sobre validación."""

    # 1. CARGAR LOS DATASETS

    # Cargamos los ejemplos de entrenamiento
    train = cargar_jsonl(TRAIN_PATH)

    # Cargamos las incidencias de validación
    validation = cargar_jsonl(VALIDATION_PATH)

    # Cargamos las 30 incidencias reservadas para la evaluación final
    test = cargar_jsonl(TEST_PATH)

    # Seleccionamos cuatro ejemplos del entrenamiento
    ejemplos = seleccionar_ejemplos(train)

    print(f"Ejemplos Few-shot seleccionados: {len(ejemplos)}")

    # Mostramos las etiquetas de los ejemplos Few-shot
    print("\nEjemplos seleccionados:")

    for numero, ejemplo in enumerate(ejemplos, start=1):

        # Mostramos la categoría y prioridad de cada ejemplo
        print(
            f"{numero}. Categoría: {ejemplo['categoria']} | "
            f"Prioridad: {ejemplo['prioridad']}"
        )

    # 2. CARGAR QWEN

    print("\nCargando Qwen3-0.6B...")

    # Cargamos el tokenizador del modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Cargamos el modelo con precisión de 16 bits
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    )

    # Movemos el modelo a la GPU (si está disponible)
    model.to(DEVICE) # type: ignore

    # Activamos el modo evaluación
    model.eval()


    # 3. PREPARAR Y EJECUTAR FEW-SHOT

    # Creamos el generador con los cuatro ejemplos seleccionados
    generador = crear_generador_few_shot(ejemplos)

    # Evaluamos Few-shot sobre las mismas incidencias del Zero-shot
    print("\nEvaluando Few-shot sobre TEST...")

    resultados, metricas = evaluar(
        test,                       # Conjunto de prueba definitivo
        model,                      # Modelo Qwen3-0.6B
        tokenizer,                  # Tokenizador del modelo
        funcion_generar=generador   # Generador con los 4 ejemplos Few-shot
    )


    # 4. GUARDAR LOS RESULTADOS

    # Guardamos las predicciones originales de Qwen
    with open(RESULTADOS_DIR / "few_shot.jsonl",
              "w", encoding="utf-8") as archivo:

        for resultado in resultados:
            archivo.write(
                json.dumps(resultado, ensure_ascii=False) + "\n"
            )

    # Guardamos las métricas de validación
    with open(RESULTADOS_DIR / "few_shot_metricas.json",
              "w", encoding="utf-8") as archivo:

        json.dump(metricas, archivo, ensure_ascii=False, indent=4)


    # 5. MOSTRAR LAS MÉTRICAS

    print("\n===== RESULTADOS FEW-SHOT (TEST) =====")

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor}%")


# Ejecutamos main() únicamente si abrimos este archivo directamente
if __name__ == "__main__":
    main()
