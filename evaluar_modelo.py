import json  # Permite leer y guardar datos en formato JSON
from pathlib import Path  # Facilita trabajar con rutas y archivos

import torch  # Librería para ejecutar modelos en CPU o GPU
from transformers import AutoTokenizer, AutoModelForCausalLM  # Carga Qwen

# Identificador del modelo que utilizaremos
MODEL_ID = "Qwen/Qwen3-0.6B"

# Seleccionamos la GPU si está disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ruta del dataset que utilizaremos para evaluar
TEST_PATH = Path("data/test.jsonl")

# Carpeta donde guardaremos los resultados
RESULTADOS_DIR = Path("resultados")

# Creamos la carpeta si todavía no existe
RESULTADOS_DIR.mkdir(exist_ok=True)

# Instrucciones que recibirá Qwen para clasificar incidencias
INSTRUCCIONES = """
Eres un asistente de soporte técnico.
Clasifica la incidencia en: acceso, software, hardware o red.
Asigna una prioridad: baja, media o alta.
Devuelve únicamente un JSON con las claves:
categoria, prioridad y respuesta.
La respuesta debe ser breve y estar en español.
"""

def cargar_jsonl(ruta):
    """Lee un archivo jsonl y devuelve sus ejemplos como una lista"""

    #Abrimos el archivo en modo lectura y con codificacion UTF-8
    with open(ruta, "r", encoding="utf-8") as archivo:

        #Convertimos cada linea json en un diccionario de python
        ejemplos = [json.loads(linea) for linea in archivo]

    return ejemplos  #Devolvemos los ejemplos cargados

def generar_respuesta(model, tokenizer, incidencia):
    """Genera una respuesta de Qwen para una incidencia"""

    #Creamos la conversacion con las instruccines y la incidencia
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES},
        {"role": "user", "content": incidencia}
    ]

    # Convertimos los mensajes al formato que entiende Qwen
    entradas = tokenizer.apply_chat_template(
        mensajes,                    # Conversación que queremos procesar
        tokenize=True,               # Convierte el texto en tokens
        add_generation_prompt=True,  # Indica dónde empieza la respuesta
        enable_thinking=False,       # Desactiva el razonamiento explícito
        return_tensors="pt",         # Devuelve tensores de PyTorch
        return_dict=True             # Devuelve un diccionario de tensores
    ).to(DEVICE)                     # Movemos los tensores a la GPU

    # Generamos la respuesta sin calcular gradientes
    with torch.inference_mode():
        salidas = model.generate(
            **entradas,                            # Pasamos los tensores al modelo
            max_new_tokens=128,                    # Máximo de tokens generados
            do_sample=False,                      # Generación determinista
            pad_token_id=tokenizer.eos_token_id    # Token utilizado para el relleno
        )

    #Eliminamos los tokens correspondientes a la pregunta
    tokens_respuesta = salidas[0][entradas["input_ids"].shape[-1]:]

    # Convertimos los tokens generados en texto legible
    respuesta = tokenizer.decode(
        tokens_respuesta,
        skip_special_tokens=True  # Ocultamos tokens especiales
    )

    # Devolvemos la respuesta original, sin modificarla
    return respuesta

def evaluar(test, model, tokenizer, funcion_generar=None):
    """Evalua las predicciones y calcula las metricas del modelo"""

    # Si no indicamos otra función, usamos la generación Zero-shot
    if funcion_generar is None:
        funcion_generar = generar_respuesta

    resultados = []  #Lista donde guardaremos cada prediccion

    #Contadores para calcular las metricas
    json_validos = 0
    categorias_correctas = 0
    prioridades_correctas = 0

    #Recorremos todas las incidencias del dataset de prueba
    for numero, ejemplo in enumerate(test, start = 1):

        #Pedimos a Qwen que clasifique la incidencia actual
        respuesta = funcion_generar(
            model, tokenizer, ejemplo["incidencia"]
        )

        #Intentamos convertir la respuesta en un objeto JSON
        try:
            prediccion = json.loads(respuesta)

            #Verificamos la estructura y los tipos del json
            valido = (
                isinstance(prediccion, dict)  #Debe ser un objeto json
                and set(prediccion.keys()) == {
                    "categoria", "prioridad", "respuesta"
                }  #Debe conener exactamente estas tres claves
            and all(
                isinstance(valor, str)
                for valor in prediccion.values()
            )  #Todos los valores deben ser cadenas de texto
        )

        except json.JSONDecodeError:
            prediccion = {}  # No podemos extraer etiquetas
            valido = False  # La salida no es un JSON válido

        #Comprobamos si la categoria coincide con la correcta
        categoria_ok = (
            valido
            and prediccion["categoria"] == ejemplo["categoria"]
        )

        # Comprobamos si la prioridad coincide con la correcta
        prioridad_ok = (
            valido
            and prediccion["prioridad"] == ejemplo["prioridad"]
        )

        #Actualizamos los contadores (True equivale a 1)
        json_validos += int(valido)
        categorias_correctas += int(categoria_ok)
        prioridades_correctas += int(prioridad_ok)

        #Guardamos la incidencia y su evaluacion
        resultados.append({
            "incidencia": ejemplo["incidencia"],
            "categoria_real": ejemplo["categoria"],
            "prioridad_real": ejemplo["prioridad"],
            "salida_original": respuesta,
            "json_valido": valido,
            "categoria_correcta": categoria_ok,
            "prioridad_correcta": prioridad_ok
        })

        #Mostramos el progreso de la evaluacion
        print(f"Evaluada incidencia {numero}/{len(test)}")

    # Número total de incidencias evaluadas
    total = len(test)

    #Calculamos los porcentajes de acierto
    metricas = {
        "json_valido_pct": round(json_validos / total * 100, 2),
        "categria_accuracy_pct": round(
            categorias_correctas / total * 100, 2
        ),
        "prioridades_accuracy_pct": round(
            prioridades_correctas / total * 100, 2
        )
    }

    return resultados, metricas

def main():
    """Carga Qwen, evalua las incidencias y guarda los resulados"""

    #1. CARGAR EL MODELO
    print(f"Modelo: {MODEL_ID}")  # Mostramos el modelo seleccionado
    print(f"Dispositivo: {DEVICE}")  # Mostramos si usamos GPU o CPU

    #Cargamos el tokenizador de Qwen
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    #Cargamos los pesos del modelo con la precision adecuada
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model.to(DEVICE)  # type: ignore
    model.eval()  # Activamos el modo evaluación

    #2. CARGAR Y EVALUAR LAS INCIDENCIAS
    test = cargar_jsonl(TEST_PATH)  #Leemos las 30 incidencias
    print(f"\nEvaluando {len(test)} incidencias...\n")

    #Ejecutamos la evaluacion y recogemos sus resultados
    resultados, metricas = evaluar(test, model, tokenizer)

    #3. GUARDAMOS LAS RESPUESTAS ORIGINALES

    #Definimos el archivo donde guardaremos cada prediccion
    ruta_resultados = RESULTADOS_DIR / "zero_short.jsonl"

    #Abrimos el archivo en modo escritura
    with open(ruta_resultados, "w", encoding="utf-8") as archivo:

        #Recorremos las predicciones generadas
        for resultado in resultados:

            #Guardamos cada resultado como una linea JSON
            archivo.write(
                json.dumps(resultado, ensure_ascii=False) + "\n"  #Convierte un objeto Python en una cadena JSON
            )

    #4. GUARDAMOS LAS METRCIAS

    #Definicmos el archivo para los porcentajes finales
    ruta_metricas = RESULTADOS_DIR / "zero_shot_metricas.json"

    #Abrimos el archivo en modo escritura
    with open(ruta_metricas, "w", encoding="utf-8") as archivo:

        #Guardamos las metricas con formato legible
        json.dump(metricas, archivo, indent=4, ensure_ascii=False)  #Escribe directamente un objeto Python como JSON en un archivo

    # 5. MOSTRAR LOS RESULTADOS

    print("\n===== RESULTADOS ZERO-SHOT =====")

    # Recorremos las métricas y mostramos sus porcentajes
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor}%")

    print("\nResultados guardados en la carpeta 'resultados'.")

# Ejecutamos main() únicamente cuando lanzamos este archivo directamente
if __name__ == "__main__":
    main()