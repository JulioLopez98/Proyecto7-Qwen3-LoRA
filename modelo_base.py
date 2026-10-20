import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Identificador del modelo en Hugging Face
OPEN_MODEL_ID = os.getenv("OPEN_MODEL_ID", "Qwen/Qwen3-0.6B")

#Seleccionamos la GPU si CUDA esta disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Modelo: {OPEN_MODEL_ID}")
print(f"Dispositivo: {DEVICE}")

#Cargamos el tokenizador del modelo
tokenizer = AutoTokenizer.from_pretrained(OPEN_MODEL_ID)

# Cargamos Qwen con precisión de 16 bits si tenemos GPU
model = AutoModelForCausalLM.from_pretrained(
    OPEN_MODEL_ID,
    dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

# Movemos el modelo al dispositivo seleccionado
model.to(device=DEVICE) # type: ignore

# Activamos el modo evaluación
model.eval()

# Instrucciones para nuestro asistente de soporte técnico
instrucciones = """
Eres un asistente de soporte técnico.
Clasifica la incidencia en: acceso, software, hardware o red.
Asigna una prioridad: baja, media o alta.
Devuelve únicamente un JSON con las claves:
categoria, prioridad y respuesta.
La respuesta debe ser breve y estar en español.
"""

# Incidencia que queremos analizar
incidencia = "Toda la oficina se ha quedado sin conexión a Internet."

#Creamos los mensajes de la conversacion
mensajes = [
    {"role": "system", "content": instrucciones},
    {"role": "user", "content": incidencia}
]

# Convertimos la conversación al formato de Qwen
entradas = tokenizer.apply_chat_template(
    mensajes,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True
).to(DEVICE)

# Generamos la respuesta sin calcular gradientes
with torch.inference_mode():
    salidas = model.generate( # type: ignore
        **entradas,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Extraemos únicamente los tokens nuevos
tokens_respuesta = salidas[0][entradas["input_ids"].shape[-1]:]

#Convertimos los tokens a texto
respuesta = tokenizer.decode(
    tokens_respuesta,
    skip_special_tokens=True
)

print("\nRESPUESTA DE QWEN:")
print(respuesta)