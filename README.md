# Proyecto 7 · Fine-tuning de Qwen3-0.6B con LoRA para soporte técnico

**Máster AI Engineer · Modelos Open Source**

Este proyecto estudia cómo adaptar un modelo de lenguaje abierto y pequeño a una tarea concreta: **clasificar incidencias de soporte técnico en español**, asignarles una prioridad y producir una respuesta breve en formato JSON.

El trabajo compara tres enfoques —**Zero-shot, Few-shot y fine-tuning con LoRA**— utilizando un conjunto de prueba común. Incluye los datos sintéticos separados, los scripts, las predicciones originales, las métricas y la publicación del adaptador entrenado en Hugging Face Hub.

> [!IMPORTANT]
> **🤗 ADAPTADOR LORA PUBLICADO EN HUGGING FACE**  
> ### [**JulioLopzz/qwen3-0.6b-soporte-ti-lora → Abrir modelo**](https://huggingface.co/JulioLopzz/qwen3-0.6b-soporte-ti-lora)
>
> El repositorio de Hugging Face contiene el adaptador LoRA, su configuración, el tokenizador y una **Model Card** detallada. Los pesos completos del modelo base **no** se incluyen en este repositorio: se cargan desde `Qwen/Qwen3-0.6B`.

---

## Índice

1. [Objetivo y planteamiento](#1-objetivo-y-planteamiento)
2. [Qué hace el asistente](#2-qué-hace-el-asistente)
3. [Tecnologías y entorno](#3-tecnologías-y-entorno)
4. [Estructura del proyecto](#4-estructura-del-proyecto)
5. [Dataset sintético y separación de datos](#5-dataset-sintético-y-separación-de-datos)
6. [Metodología de evaluación](#6-metodología-de-evaluación)
7. [Experimento 1: Zero-shot](#7-experimento-1-zero-shot)
8. [Experimento 2: Few-shot](#8-experimento-2-few-shot)
9. [Experimento 3: fine-tuning con LoRA](#9-experimento-3-fine-tuning-con-lora)
10. [Resultados y comparación](#10-resultados-y-comparación)
11. [Cómo instalar y reproducir el proyecto](#11-cómo-instalar-y-reproducir-el-proyecto)
12. [Publicación y recarga desde Hugging Face Hub](#12-publicación-y-recarga-desde-hugging-face-hub)
13. [Limitaciones y posibles mejoras](#13-limitaciones-y-posibles-mejoras)
14. [Conclusiones](#14-conclusiones)

---

## 1. Objetivo y planteamiento

En un servicio de soporte TI, una incidencia escrita en lenguaje natural debe interpretarse y clasificarse antes de decidir cómo gestionarla. Por ejemplo, «No puedo iniciar sesión» apunta a un problema de acceso, mientras que «No funciona la conexión wifi» apunta a un problema de red.

El objetivo es investigar si un modelo abierto de **0,6 mil millones de parámetros** puede resolver esta tarea sin entrenamiento adicional y cuánto mejora al adaptarlo con ejemplos propios.

Se plantean tres alternativas:

| Enfoque | ¿Qué recibe el modelo? | ¿Se modifican sus pesos? |
|---|---|---|
| **Zero-shot** | Instrucciones de la tarea, sin ejemplos | No |
| **Few-shot** | Las mismas instrucciones y algunos ejemplos de referencia | No |
| **LoRA** | Las instrucciones y un adaptador entrenado con incidencias etiquetadas | Sí, únicamente los parámetros del adaptador |

La comparación permite responder a una pregunta práctica: **¿cuándo basta con mejorar el prompt y cuándo compensa entrenar un adaptador?**

### Alcance

Este es un **prototipo educativo** de clasificación y respuesta, no una herramienta de gestión de tickets en producción. No abre tickets, no accede a sistemas empresariales y no ejecuta acciones sobre equipos o redes.

## 2. Qué hace el asistente

Recibe una incidencia en español y debe devolver **exclusivamente un objeto JSON** con tres claves:

```json
{
  "categoria": "red",
  "prioridad": "alta",
  "respuesta": "Comprueba la conexión y contacta con soporte."
}
```

### Campos de salida

| Campo | Valores esperados | Significado |
|---|---|---|
| `categoria` | `acceso`, `software`, `hardware`, `red` | Tipo de incidencia |
| `prioridad` | `baja`, `media`, `alta` | Nivel de urgencia estimado |
| `respuesta` | Texto breve en español | Primera orientación para el usuario |

Las cuatro categorías se interpretan así:

- **Acceso:** inicio de sesión, cuentas, contraseñas y permisos.
- **Software:** aplicaciones, programas y errores de funcionamiento.
- **Hardware:** ordenadores, periféricos y componentes físicos.
- **Red:** conectividad, wifi y acceso a recursos de red.

La prioridad depende del **impacto y la urgencia** descritos en la incidencia. Este punto es especialmente difícil cuando el mensaje no indica cuántas personas están afectadas o si existe una interrupción crítica.

### Instrucción utilizada

Se mantiene una instrucción común para los experimentos:

```text
Eres un asistente de soporte técnico.
Clasifica la incidencia en: acceso, software, hardware o red.
Asigna una prioridad: baja, media o alta.
Devuelve únicamente un JSON con las claves:
categoria, prioridad y respuesta.
La respuesta debe ser breve y estar en español.
```

La generación se realiza de forma **determinista** (`do_sample=False`), con un límite de **128 tokens nuevos** y el modo de razonamiento de Qwen3 desactivado para la generación (`enable_thinking=False`). Esto reduce diferencias debidas al muestreo y facilita la comparación.

## 3. Tecnologías y entorno

| Componente | Elección | Función |
|---|---|---|
| Modelo base | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | Modelo de lenguaje de partida |
| Python | 3.10.11 | Lenguaje de programación |
| PyTorch | 2.11.0 + CUDA 12.8 | Ejecución y entrenamiento en GPU |
| Transformers | 5.20.1 | Modelo, tokenizador y generación |
| Datasets | 5.1.0 | Gestión de datos para entrenamiento |
| TRL | 1.14.2 | Entrenamiento supervisado mediante `SFTTrainer` |
| PEFT | 0.21.2 | Configuración, guardado y carga de LoRA |
| Accelerate | 1.15.0 | Soporte para entrenamiento |
| Hugging Face Hub | 2.2.0 | Publicación y descarga del adaptador |

**Equipo utilizado:** Windows, NVIDIA RTX 4070 SUPER de **12 GB de VRAM** y **32 GB de RAM**.

> Las versiones reflejan el entorno con el que se realizó el experimento. La disponibilidad de GPU CUDA depende del equipo del usuario. Para repetir el entrenamiento con una configuración distinta, pueden ser necesarios ajustes de memoria o precisión.

## 4. Estructura del proyecto

```text
Proyecto7-OpenSource/
│
├── data/
│   ├── train.jsonl                 # Datos de entrenamiento
│   ├── validation.jsonl            # Datos de validación
│   └── test.jsonl                  # Datos de prueba finales
│
├── resultados/
│   ├── zero_shot.jsonl             # Predicciones originales Zero-shot
│   ├── zero_shot_metricas.json     # Métricas Zero-shot
│   ├── few_shot.jsonl              # Predicciones originales Few-shot
│   ├── few_shot_metricas.json      # Métricas Few-shot
│   ├── lora.jsonl                  # Predicciones originales LoRA
│   └── lora_metricas.json          # Métricas LoRA
│
├── generar_datos.py                # Generación del dataset sintético
├── modelo_base.py                  # Carga / prueba del modelo base
├── evaluar_modelo.py               # Funciones comunes de generación y evaluación
├── evaluar_few_shot.py             # Evaluación con ejemplos en el prompt
├── entrenar_lora.py                # Entrenamiento supervisado del adaptador
├── evaluar_lora.py                 # Evaluación del adaptador entrenado
├── publicar_lora.py                # Publicación en Hugging Face Hub
├── probar_modelo_hub.py            # Recarga y prueba desde el Hub
├── requirements.txt               # Dependencias principales
├── .gitignore                      # Exclusiones de Git
└── README.md                       # Esta documentación
```

**Nota sobre los archivos locales:** durante el entrenamiento se crea `modelo_lora/`, que contiene checkpoints y el adaptador final. Esta carpeta se excluye de GitHub para evitar duplicar archivos de pesos; el adaptador final está disponible públicamente en [Hugging Face](https://huggingface.co/JulioLopzz/qwen3-0.6b-soporte-ti-lora).

## 5. Dataset sintético y separación de datos

El dataset se genera con `generar_datos.py`. Contiene **260 incidencias sintéticas en español**, elaboradas a partir de **24 escenarios base** y sus variantes.

Cada ejemplo representa una incidencia con su categoría, prioridad y respuesta esperada. Los archivos se almacenan en formato **JSONL** (*JSON Lines*): una incidencia JSON por línea. Este formato permite leer y procesar ejemplos de forma individual.

### División utilizada

| Conjunto | Archivo | Ejemplos | Uso |
|---|---|---:|---|
| Entrenamiento | `data/train.jsonl` | **200** | Aprendizaje del adaptador LoRA |
| Validación | `data/validation.jsonl` | **30** | Comprobaciones durante el desarrollo y seguimiento del entrenamiento |
| Test | `data/test.jsonl` | **30** | Evaluación final de los tres enfoques |
| **Total** | | **260** | |

La separación se establece **antes de comparar los métodos**. El conjunto de test se mantiene fijo y no se utiliza para entrenar el adaptador ni para seleccionar los ejemplos del prompt Few-shot.

En Few-shot, los ejemplos de referencia proceden del conjunto de entrenamiento y se eligieron utilizando validación, no test.

### Distribución del conjunto de test

**Por categoría:**

| Categoría | Ejemplos |
|---|---:|
| Acceso | 7 |
| Software | 9 |
| Hardware | 5 |
| Red | 9 |
| **Total** | **30** |

**Por prioridad:**

| Prioridad | Ejemplos |
|---|---:|
| Baja | 10 |
| Media | 12 |
| Alta | 8 |
| **Total** | **30** |

### Por qué importa separar los datos

- **Train:** permite ajustar el modelo a la tarea.
- **Validation:** ayuda a observar el aprendizaje y tomar decisiones durante el desarrollo.
- **Test:** sirve para estimar el rendimiento final sobre ejemplos no empleados en el entrenamiento ni en la selección del método.

Usar los mismos ejemplos para entrenar y evaluar produciría resultados engañosos. La separación mejora la **validez de la comparación**, aunque no elimina por completo las limitaciones de un dataset pequeño y sintético.

> **Importante:** los ejemplos proceden de escenarios sintéticos y variantes. Aunque los registros se hayan separado, pueden compartir patrones lingüísticos o temáticos. Por ello, el resultado no equivale a una validación con tickets reales de una empresa.

## 6. Metodología de evaluación

Los tres enfoques se evalúan con el **mismo archivo `data/test.jsonl` (30 ejemplos)** y las mismas reglas generales de generación y puntuación.

La lógica común está en `evaluar_modelo.py`, lo que evita crear evaluadores diferentes que puedan favorecer involuntariamente a un método.

### Métricas

1. **JSON válido (%):** proporción de respuestas que se pueden interpretar como JSON y contienen exactamente las claves `categoria`, `prioridad` y `respuesta`, con valores de tipo texto.
2. **Exactitud de categoría (%):** proporción de ejemplos cuya categoría predicha coincide con la categoría esperada.
3. **Exactitud de prioridad (%):** proporción de ejemplos cuya prioridad predicha coincide con la prioridad esperada.

Las predicciones con JSON inválido se contabilizan como incorrectas para las métricas de clasificación. **No se reparan automáticamente** ni se repite la generación para conseguir una salida mejor.

Las métricas de categoría y prioridad son independientes: una incidencia puede tener la categoría correcta y la prioridad incorrecta.

### Condiciones de comparación

| Aspecto | Criterio |
|---|---|
| Modelo de partida | Qwen3-0.6B |
| Idioma | Español |
| Conjunto de test | Los mismos 30 ejemplos |
| Estructura de salida | Las mismas tres claves JSON |
| Muestreo | Desactivado (`do_sample=False`) |
| Longitud máxima | 128 tokens nuevos |
| Corrección automática de respuestas | No |
| Diferencia entre métodos | Instrucciones sin ejemplos, ejemplos en el prompt o pesos adaptados mediante LoRA |

La comparación es justa **dentro del alcance de este experimento**: se mantienen el test y las reglas de puntuación. No significa que los tres enfoques tengan el mismo coste de preparación, entrenamiento o inferencia.

## 7. Experimento 1: Zero-shot

### ¿En qué consiste?

En **Zero-shot** se utiliza el modelo base sin modificar sus pesos y sin añadir ejemplos resueltos al prompt. Solo se le indica qué tarea debe realizar y qué formato debe devolver.

Este experimento responde a la pregunta: **¿qué sabe hacer Qwen3-0.6B antes de adaptarlo?**

### Resultado

- **JSON válido:** 100 % (**30/30**).
- **Categoría correcta:** 73,33 % (**22/30**).
- **Prioridad correcta:** 30,00 % (**9/30**).

**Interpretación:** el modelo entiende bien el formato exigido y clasifica correctamente una parte importante de las incidencias, pero tiene dificultades considerables para estimar la prioridad.

**Archivos originales:** `resultados/zero_shot.jsonl` y `resultados/zero_shot_metricas.json`.

## 8. Experimento 2: Few-shot

### ¿En qué consiste?

En **Few-shot** se conserva el modelo base, pero se incluyen **cuatro ejemplos resueltos** como referencia en el contexto de la conversación.

Estos ejemplos ayudan a mostrarle al modelo cómo asociar descripciones de incidencias con categorías, prioridades y respuestas. **No existe entrenamiento de pesos**: el cambio ocurre únicamente en el prompt.

Los ejemplos se seleccionaron a partir del conjunto de entrenamiento, utilizando validación durante el desarrollo. No se utilizaron las etiquetas del test para escogerlos.

### Resultado

- **JSON válido:** 100 % (**30/30**).
- **Categoría correcta:** 83,33 % (**25/30**).
- **Prioridad correcta:** 43,33 % (**13/30**).

**Interpretación:** añadir ejemplos mejora el rendimiento frente a Zero-shot sin necesidad de entrenar. La prioridad continúa siendo la tarea más difícil.

**Archivos originales:** `resultados/few_shot.jsonl` y `resultados/few_shot_metricas.json`.

## 9. Experimento 3: fine-tuning con LoRA

### ¿Qué es LoRA y por qué se utiliza?

**LoRA** (*Low-Rank Adaptation*) es una técnica de **PEFT** (*Parameter-Efficient Fine-Tuning*). En lugar de actualizar todos los parámetros del modelo base, introduce matrices pequeñas y entrenables en determinadas capas.

En este proyecto se utiliza para adaptar Qwen3-0.6B a la clasificación de incidencias sin tener que reentrenar sus aproximadamente 597 millones de parámetros.

**Ventajas prácticas:**

- Reduce considerablemente el número de parámetros que deben entrenarse.
- Permite trabajar con una GPU de consumo.
- Genera un adaptador pequeño, separado del modelo base.
- Facilita compartir el ajuste en Hugging Face sin publicar una copia completa del modelo original.

### Configuración LoRA

| Parámetro | Valor | Significado |
|---|---|---|
| `r` | `8` | Rango de las matrices LoRA |
| `lora_alpha` | `16` | Escalado de la actualización LoRA |
| `lora_dropout` | `0.05` | Regularización durante entrenamiento |
| `target_modules` | `q_proj`, `v_proj` | Proyecciones de atención adaptadas |
| `bias` | `none` | No se entrenan sesgos adicionales |
| `task_type` | `CAUSAL_LM` | Modelo de lenguaje autorregresivo |

**Parámetros entrenables:** **1.146.880 de 597.196.800**, aproximadamente **0,192 %** del total.

Este dato es importante: el entrenamiento adapta una fracción muy pequeña del modelo y mantiene congelados los parámetros base.

### Entrenamiento supervisado

Se utiliza `SFTTrainer` de TRL y ejemplos con formato de conversación.

| Hiperparámetro | Valor |
|---|---|
| Épocas | **3** |
| Batch size por dispositivo | `2` |
| Acumulación de gradientes | `4` |
| Batch efectivo aproximado | `8` ejemplos |
| Learning rate | `2e-4` |
| Precisión | `fp16` |
| Longitud máxima | `512` tokens |
| `assistant_only_loss` | `True` |
| Evaluación | Al final de cada época |
| Guardado de checkpoints | Al final de cada época |
| Pasos de entrenamiento | **75** |
| Tiempo observado | Aproximadamente **127 segundos** |

**¿Qué significa `assistant_only_loss=True`?** Durante el entrenamiento, la función de pérdida se aplica a la respuesta del asistente, no a los tokens del sistema ni del usuario. Así, el aprendizaje se centra en generar la clasificación y respuesta esperadas.

En una comprobación de ejemplo, **37 de 135 tokens** participaban en la pérdida; los demás estaban enmascarados.

### Evolución de la pérdida de validación

| Época | `eval_loss` |
|---|---:|
| 1 | 0,6598 |
| 2 | 0,6679 |
| 3 | 0,7240 |

La pérdida de validación **aumenta** después de la primera época. Esto sugiere que el entrenamiento adicional podría estar empezando a perjudicar la generalización o que existe variabilidad relevante en un conjunto de validación pequeño.

**Decisión del experimento:** se publica y evalúa el **adaptador final de tres épocas**. Los resultados de test no deben utilizarse para escoger retrospectivamente un checkpoint. Una mejora futura sería comparar checkpoints **solo en validación** y evaluar una única vez el seleccionado sobre test.

### Resultado del adaptador final

- **JSON válido:** 100 % (**30/30**).
- **Categoría correcta:** 100 % (**30/30**).
- **Prioridad correcta:** 53,33 % (**16/30**).

**Archivos originales:** `resultados/lora.jsonl` y `resultados/lora_metricas.json`.

## 10. Resultados y comparación

### Tabla comparativa final

| Método | JSON válido | Categoría correcta | Prioridad correcta |
|---|---:|---:|---:|
| **Zero-shot** | 100,00 % | 73,33 % (22/30) | 30,00 % (9/30) |
| **Few-shot** | 100,00 % | 83,33 % (25/30) | 43,33 % (13/30) |
| **LoRA (3 épocas)** | **100,00 %** | **100,00 % (30/30)** | **53,33 % (16/30)** |

### ¿Cuánto mejora LoRA?

Frente a **Zero-shot**:

- **+26,67 puntos porcentuales** en categoría.
- **+23,33 puntos porcentuales** en prioridad.

Frente a **Few-shot**:

- **+16,67 puntos porcentuales** en categoría.
- **+10,00 puntos porcentuales** en prioridad.

### Lectura crítica

1. **El formato JSON no diferencia a los métodos:** los tres alcanzan el 100 % de respuestas válidas en el test.
2. **LoRA obtiene la mejor clasificación de categorías:** acierta las 30 incidencias del test.
3. **La prioridad sigue siendo mejorable:** LoRA acierta 16 de 30 casos, de modo que no debe considerarse un clasificador fiable de urgencias sin validación adicional.
4. **La muestra es pequeña:** cada ejemplo representa aproximadamente 3,33 puntos porcentuales. Las diferencias observadas no garantizan el mismo comportamiento en datos reales.
5. **La métrica no evalúa la utilidad de la respuesta textual:** se comprueba que exista una respuesta en español y formato correcto, pero no se dispone de una evaluación humana sistemática de su calidad técnica.

### Evidencias originales

Las predicciones completas y las métricas se incluyen en `resultados/` para que cualquier persona pueda revisar los aciertos y errores, no solo los porcentajes agregados.

## 11. Cómo instalar y reproducir el proyecto

### 11.1. Requisitos previos

- Python **3.10** (versión utilizada: 3.10.11).
- Git.
- Conexión a Internet para descargar Qwen3-0.6B y, cuando corresponda, el adaptador.
- Para reproducir el entrenamiento con la configuración original: una GPU NVIDIA compatible con CUDA y memoria suficiente. El experimento se realizó con **12 GB de VRAM**.

### 11.2. Clonar el repositorio

```powershell
git clone <URL_DEL_REPOSITORIO_GITHUB>
cd Proyecto7-OpenSource
```

> **Pendiente al publicar en GitHub:** sustituir `<URL_DEL_REPOSITORIO_GITHUB>` por la URL real del repositorio. No se incluye una URL inventada.

### 11.3. Crear y activar el entorno virtual (Windows / PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 11.4. Instalar PyTorch

En el equipo utilizado se trabajó con **PyTorch 2.11.0 + CUDA 12.8**. Para reproducir esa compilación en un equipo compatible, se puede instalar desde el índice oficial de PyTorch:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Si no se dispone de GPU NVIDIA, hay que elegir la instalación de PyTorch apropiada para el equipo desde [la web oficial de PyTorch](https://pytorch.org/get-started/locally/). La ejecución en CPU puede ser significativamente más lenta y el script de entrenamiento original está configurado para `fp16`.

### 11.5. Instalar las dependencias del proyecto

```powershell
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las **dependencias principales con versiones fijadas**. No es un bloqueo completo de todas las dependencias transitivas: una instalación futura puede resolver versiones indirectas diferentes. Para una reproducción idéntica a nivel de paquetes se necesitaría, además, un archivo de bloqueo del entorno completo.

### 11.6. Comprobar que PyTorch reconoce la GPU

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

Si muestra `CUDA disponible: True`, PyTorch reconoce una GPU compatible.

### 11.7. Datos sintéticos

Los tres archivos del dataset **ya se incluyen en el repositorio**. No hace falta regenerarlos para revisar las evaluaciones originales.

Para estudiar o repetir el proceso de creación de datos, consulta `generar_datos.py`. **Atención:** regenerar los archivos puede sobrescribir la separación utilizada en los resultados publicados. Conserva una copia de `data/` antes de ejecutar el generador y no compares resultados nuevos con las métricas antiguas como si se tratara del mismo test.

### 11.8. Evaluar el modelo base

La lógica de generación y evaluación común está en `evaluar_modelo.py`. El proyecto incluye `modelo_base.py` para trabajar con el modelo original y `evaluar_few_shot.py` para el experimento con ejemplos.

```powershell
python modelo_base.py
python evaluar_few_shot.py
```

> Los comandos anteriores ejecutan los scripts correspondientes. Para interpretar exactamente qué archivos crean o actualizan, revisa sus bloques `if __name__ == "__main__"` y las rutas definidas en cada script. Las métricas originales entregadas se conservan en `resultados/`.

### 11.9. Repetir el entrenamiento LoRA

```powershell
python entrenar_lora.py
```

Este script configura LoRA, prepara los datos, ejecuta el entrenamiento supervisado y guarda el adaptador final en una carpeta local dentro de `modelo_lora/`.

**Importante para la reproducibilidad:** repetir un entrenamiento no garantiza obtener pesos o porcentajes idénticos si cambian la semilla aleatoria, las dependencias, la GPU o el comportamiento numérico. La configuración y los resultados originales permiten **repetir el procedimiento** y comparar el resultado, no prometen identidad bit a bit.

### 11.10. Evaluar el adaptador entrenado

```powershell
python evaluar_lora.py
```

El script carga el modelo base y el adaptador local, ejecuta la evaluación sobre `data/test.jsonl` y guarda las predicciones y métricas correspondientes.

**Precaución:** si vuelves a ejecutar los scripts, comprueba sus rutas de salida para no sobrescribir accidentalmente los resultados originales incluidos en la entrega.

## 12. Publicación y recarga desde Hugging Face Hub

### 12.1. ¿Qué se ha publicado?

El adaptador entrenado está disponible públicamente en:

### **🤗 [ABRIR ADAPTADOR QWEN3-0.6B SOPORTE TI LORA EN HUGGING FACE](https://huggingface.co/JulioLopzz/qwen3-0.6b-soporte-ti-lora)**

**ID del repositorio:** `JulioLopzz/qwen3-0.6b-soporte-ti-lora`

En el Hub se verificó la presencia de estos archivos:

```text
.gitattributes
README.md
adapter_config.json
adapter_model.safetensors
chat_template.jinja
tokenizer.json
tokenizer_config.json
training_args.bin
```

- `adapter_model.safetensors`: pesos entrenados del adaptador.
- `adapter_config.json`: configuración que indica cómo aplicar LoRA.
- `tokenizer.json`, `tokenizer_config.json` y `chat_template.jinja`: archivos para tokenizar y preparar correctamente los mensajes.
- `README.md`: **Model Card** con descripción, entrenamiento, evaluación y limitaciones.
- `training_args.bin`: configuración de entrenamiento guardada por el proceso.

El repositorio contiene **el adaptador**, no una copia íntegra de Qwen3-0.6B. Para utilizarlo se necesita cargar también el modelo base.

### 12.2. Publicar el adaptador desde el entorno local

El script `publicar_lora.py` crea el repositorio público si no existe y sube el contenido de `modelo_lora/adapter_final/`.

```powershell
python publicar_lora.py
```

**No es necesario ejecutar este paso para usar el modelo ya publicado.** Solo es útil si se vuelve a entrenar y se desea publicar una nueva versión. Publicar requiere autenticación y permisos de escritura en Hugging Face.

### 12.3. Recargar el modelo directamente desde el Hub

La comprobación final se realizó con `probar_modelo_hub.py`:

```powershell
python probar_modelo_hub.py
```

El script realiza estas operaciones:

1. Descarga o recupera de caché el tokenizador usando el ID público del adaptador.
2. Carga el modelo base `Qwen/Qwen3-0.6B`.
3. Obtiene y aplica el adaptador con `PeftModel.from_pretrained(modelo_base, REPO_ID)`.
4. Ejecuta una clasificación usando la misma función de generación del proyecto.

Fragmento clave:

```python
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3-0.6B"
REPO_ID = "JulioLopzz/qwen3-0.6b-soporte-ti-lora"

# Una vez cargado el modelo base:
modelo = PeftModel.from_pretrained(modelo_base, REPO_ID)
```

**Diferencia importante:** se utiliza `REPO_ID` en lugar de una ruta local como `modelo_lora/adapter_final`. Así, el adaptador puede recuperarse desde Hugging Face sin depender de los pesos guardados en el proyecto.

### 12.4. Resultado real de la prueba

**Incidencia enviada:**

```text
No puedo conectarme a la red wifi de la oficina.
```

**Respuesta obtenida:**

```json
{
  "categoria": "red",
  "prioridad": "alta",
  "respuesta": "Comprueba la conexión y contacta con soporte."
}
```

La prueba confirmó que el adaptador **se cargó por su identificador del Hub y generó un JSON válido**, con la categoría `red` correcta.

La prioridad `alta` es **discutible**: el mensaje no indica si la caída afecta a una sola persona o a toda la oficina. Este ejemplo refleja una limitación ya observada en las métricas de prioridad.

> Hugging Face puede reutilizar archivos de su caché local. Por tanto, la prueba verifica la **carga mediante un identificador remoto**, aunque no constituye por sí sola una demostración de descarga desde una máquina completamente nueva y sin caché.

## 13. Limitaciones y posibles mejoras

### Limitaciones actuales

1. **Datos sintéticos:** no representan toda la variedad, ambigüedad y ruido de incidencias reales.
2. **Conjunto de test pequeño:** 30 ejemplos ofrecen una primera referencia, no una garantía estadística de rendimiento en producción.
3. **Variantes de escenarios base:** pueden existir similitudes entre los patrones lingüísticos de distintos conjuntos, aunque los registros sean diferentes.
4. **Prioridades poco fiables:** incluso con LoRA, la exactitud es del **53,33 %**. No conviene automatizar decisiones críticas de urgencia con este resultado.
5. **Pérdida de validación creciente:** después de la primera época, `eval_loss` aumenta, lo que aconseja investigar posibles problemas de generalización.
6. **Respuesta textual no evaluada exhaustivamente:** el proyecto mide formato, categoría y prioridad, pero no la utilidad técnica de cada consejo mediante una rúbrica humana.
7. **Reproducibilidad limitada por el entorno:** dependencias indirectas, hardware y aleatoriedad pueden afectar a una nueva ejecución.

### Mejoras propuestas

- **Ampliar y diversificar los datos**, especialmente los ejemplos de prioridad, e incluir incidencias ambiguas.
- **Definir una rúbrica explícita de urgencia**, con criterios de impacto, usuarios afectados y criticidad del servicio.
- **Separar por escenarios base** al construir train, validation y test, para medir mejor la generalización a situaciones nuevas.
- **Seleccionar la mejor época utilizando solo validación**, sin consultar test durante la selección.
- **Añadir métricas por clase**, como matriz de confusión, precisión, recall y F1.
- **Evaluar la calidad de `respuesta`** mediante revisión humana y criterios técnicos.
- **Comprobar la recarga en un entorno limpio**, sin archivos del adaptador previamente descargados.
- **Guardar una configuración más estricta de reproducibilidad**, incluyendo semilla y bloqueo de dependencias transitivas.

## 14. Conclusiones

El experimento demuestra que un modelo abierto y relativamente pequeño como **Qwen3-0.6B** puede adaptarse a una tarea especializada en español mediante LoRA, entrenando aproximadamente **el 0,192 % de sus parámetros**.

Los resultados sobre el test sintético muestran una progresión clara: Zero-shot alcanza **73,33 %** en categoría, Few-shot sube a **83,33 %** y LoRA obtiene **100 %**. En prioridad también mejora, del **30,00 %** al **53,33 %**, aunque este último resultado sigue siendo insuficiente para una automatización fiable de la urgencia.

Además del entrenamiento, se completó el ciclo de trabajo: **generación y separación de datos, comparación bajo reglas comunes, conservación de resultados originales, publicación del adaptador y prueba de recarga desde Hugging Face Hub**.

**Conclusión principal:** LoRA ha sido el enfoque con mejor rendimiento en este experimento, pero el tamaño y la naturaleza sintética del dataset obligan a interpretar los resultados con prudencia. El proyecto sirve como demostración reproducible del proceso de adaptación y evaluación de un modelo open source, no como sistema listo para producción.

---

### Enlaces de referencia

- **🤗 Adaptador LoRA entrenado:** [**JulioLopzz/qwen3-0.6b-soporte-ti-lora**](https://huggingface.co/JulioLopzz/qwen3-0.6b-soporte-ti-lora)
- **Modelo base:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Transformers:** https://huggingface.co/docs/transformers
- **PEFT:** https://huggingface.co/docs/peft
- **TRL:** https://huggingface.co/docs/trl

**Proyecto académico · Máster AI Engineer · Modelos Open Source**
