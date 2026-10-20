import json #nos permitirá guardar las incidencias en formato JSON
import random #servirá para variar y mezclar los ejemplos
from pathlib import Path

#Garantiza que las operaciones aleatorias produzcan los mismos resultados cada vez que ejecutemos el script
random.seed(42)

#Carpeta donde guardaremos los datasets
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True) #crea la carpeta data si no existe

#Cantidad de ejemplos por conjunto
N_TRAIN = 200
N_VALIDACION = 30
N_TEST = 30

#Categorias y prioridades permitidas
CATEGORIAS = ["acceso", "software", "hardware", "red"]
PRIORIDADES = ["baja", "media", "alta"]

#Situaciones ficticias de soporte tecnico
#Cada entrada contiene: categoria, prioridad, problema y respuesta
INCIDENCIAS = [
    (
        "acceso",
        "baja",
        "Quiero cambiar mi contraseña por seguridad",
        "Puedes cambiarla desde la configuración de tu cuenta."
    ),
    (
        "acceso",
        "media",
        "No puedo iniciar sesión en mi cuenta",
        "Comprueba tus credenciales y restablece la contraseña."
    ),
    (
        "acceso",
        "alta",
        "Ningún empleado puede acceder al sistema",
        "Contacta inmediatamente con el equipo de soporte."
    ),
    (
        "software",
        "baja",
        "Quiero cambiar el idioma de una aplicación",
        "Revisa las preferencias de idioma de la aplicación."
    ),
    (
        "software",
        "media",
        "La aplicación se cierra algunas veces",
        "Actualiza la aplicación y comprueba si el problema continúa."
    ),
    (
        "software",
        "alta",
        "El programa principal de la empresa no funciona",
        "Comunica la incidencia urgente al equipo técnico."
    ),
    (
        "hardware",
        "baja",
        "El ratón hace ruido al pulsar",
        "Comprueba el estado del ratón y solicita su revisión."
    ),
    (
        "hardware",
        "media",
        "El teclado deja de responder ocasionalmente",
        "Revisa la conexión y prueba con otro teclado."
    ),
    (
        "hardware",
        "alta",
        "El ordenador no enciende y no puedo trabajar",
        "Comprueba la alimentación y contacta con soporte."
    ),
    (
        "red",
        "baja",
        "La conexión WiFi es algo lenta",
        "Comprueba la intensidad de la señal WiFi."
    ),
    (
        "red",
        "media",
        "Internet se desconecta varias veces al día",
        "Reinicia el router y comprueba la estabilidad de la conexión."
    ),
    (
        "red",
        "alta",
        "Toda la oficina se ha quedado sin Internet",
        "Notifica inmediatamente la caída de red al soporte técnico."
    ),

    # Nuevas incidencias para aumentar la variedad del dataset
    ("acceso", "baja",
     "Quiero actualizar mi pregunta de seguridad",
     "Accede a la configuración de seguridad de tu cuenta."),

    ("acceso", "media",
     "Mi cuenta está bloqueada después de varios intentos",
     "Solicita el desbloqueo de tu cuenta al soporte técnico."),

    ("acceso", "alta",
     "Se han bloqueado las cuentas de todos los administradores",
     "Contacta urgentemente con el equipo de seguridad."),

    ("software", "baja",
     "Quiero modificar las notificaciones del programa",
     "Revisa las opciones de notificaciones en preferencias."),

    ("software", "media",
     "El programa tarda demasiado en abrir documentos",
     "Comprueba las actualizaciones y reinicia la aplicación."),

    ("software", "alta",
     "El sistema de facturación está completamente detenido",
     "Notifica inmediatamente la incidencia al soporte técnico."),

    ("hardware", "baja",
     "La pantalla tiene una pequeña marca superficial",
     "Solicita una revisión del monitor cuando sea posible."),

    ("hardware", "media",
     "La impresora deja de funcionar de manera intermitente",
     "Comprueba las conexiones y reinicia la impresora."),

    ("hardware", "alta",
     "El servidor principal no consigue arrancar",
     "Contacta urgentemente con el equipo de infraestructura."),

    ("red", "baja",
     "Quiero consultar la velocidad de mi conexión",
     "Realiza una prueba de velocidad de la conexión."),

    ("red", "media",
     "La conexión VPN se desconecta constantemente",
     "Comprueba la configuración VPN y contacta con soporte."),

    ("red", "alta",
     "La red corporativa está completamente inaccesible",
     "Notifica urgentemente la caída al equipo de redes.")
]

#Diferentes formas de expresar una incidencia
PLANTILLAS = [
    "{}",
    "Necesito ayuda: {}",
    "Tengo el siguiente problema: {}",
    "He abierto una incidencia porque {}",
    "Desde esta mañana, {}",
    "Solicito asistencia técnica: {}",
    "Me gustaría informar de que {}",
    "Estoy teniendo problemas: {}",
    "¿Podéis ayudarme? {}",
    "Necesito soporte porque {}",
    "Os escribo para comunicar que {}",
    "Quería consultar lo siguiente: {}",
    "Tengo una incidencia relacionada con esto: {}",
    "¿Cómo puedo solucionarlo? {}",
    "Por favor, revisad este problema: {}",
    "He detectado lo siguiente: {}",
    "Necesito una solución para esto: {}",
    "Os contacto porque {}",
    "Me está ocurriendo lo siguiente: {}",
    "Quisiera solicitar ayuda: {}",
    "Hay un problema que necesito resolver: {}",
    "Estoy intentando trabajar, pero {}",
    "Necesito que reviséis esta situación: {}",
    "He observado este fallo: {}",
    "¿Podríais indicarme qué hacer? {}",
    "Tengo dificultades con lo siguiente: {}",
    "Os informo de una incidencia: {}",
    "Necesito asistencia para resolver esto: {}"
]

def generar_ejemplos(incidencias, cantidad):
    """Genera incidencias sinteticas con su clasificacion correcta a partir de incidencias y plantillas"""
    ejemplos = []
    textos_usados = set()  #Guardamos las incidencias ya generadas

    #Recorremos las 12 incidencias originales
    for categoria, prioridad, problema, respuesta in incidencias:
        #Creamos una variacion or cada plantilla
        for plantilla in PLANTILLAS:

            #Insertamos el problema dentro de la plantilla
            texto = plantilla.format(problema)

            #Si ya existe el texto, no volvemos a añadirlo
            if texto in textos_usados:
                continue

            textos_usados.add(texto)  #Marcamos el texto como utilizado
            
            #Construimos el ejemplo con su respuesta esperada
            ejemplo = {
                "incidencia": texto,
                "categoria": categoria,
                "prioridad": prioridad,
                "respuesta": respuesta
            }

            ejemplos.append(ejemplo)

    #Mezclamos los ejemplos para distribuirlos aleatoriamente
    random.shuffle(ejemplos)

    return ejemplos[:cantidad]

def guardar_jsonl(nombre_archivo, ejemplos):
    """Guarda los ejemplos en un archivo JSONL (un JSON por linea)"""

    ruta = DATA_DIR / nombre_archivo  #Construye la ruta dentro de /data

    #Abrimos el archivo en modo escritura y con soporte para tildes
    with open(ruta, "w", encoding="utf-8") as archivo:

        for ejemplo in ejemplos:  #Recorremos cada incidencia

            #Convertimos el diccionario en un JSON y añadimos un salto de linea
            archivo.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

    print (f"{nombre_archivo}: {len(ejemplos)} ejemplos")  #Confirmamos el guardado



def preparar_datasets():
    """Genera, valida y guarda los tres datasets equilibrados."""

    # 1. PREPARAR LAS INCIDENCIAS

    # Eliminamos posibles tuplas repetidas
    incidencias_unicas = list(dict.fromkeys(INCIDENCIAS))

    # Extraemos los textos originales para comprobar que sean distintos
    problemas = [incidencia[2] for incidencia in incidencias_unicas]

    assert len(incidencias_unicas) == 24, "Se esperaban 24 incidencias"
    assert len(problemas) == len(set(problemas)), "Hay problemas repetidos"


    # 2. AGRUPAR POR CATEGORÍA Y PRIORIDAD

    grupos = {}  # Diccionario con las 12 combinaciones posibles

    for incidencia in incidencias_unicas:

        # Extraemos la categoría y la prioridad
        categoria, prioridad = incidencia[0], incidencia[1]

        # Creamos una clave para identificar cada combinación
        clave = (categoria, prioridad)

        # Añadimos la incidencia al grupo correspondiente
        grupos.setdefault(clave, []).append(incidencia)

    # Comprobamos que haya 12 grupos con 2 incidencias cada uno
    assert len(grupos) == 12, "Se esperaban 12 combinaciones"
    assert all(len(grupo) == 2 for grupo in grupos.values())

    # Mezclamos las incidencias dentro de cada grupo
    for grupo in grupos.values():
        random.shuffle(grupo)


    # 3. REPARTIR LOS PROBLEMAS ORIGINALES

    train_base = []       # Problemas para entrenamiento
    validation_base = []  # Problemas para validación
    test_base = []        # Problemas para prueba

    # Recorremos las categorías y prioridades en orden fijo
    for categoria in CATEGORIAS:

        for indice, prioridad in enumerate(PRIORIDADES):

            # Recuperamos las dos incidencias de esta combinación
            primera, segunda = grupos[(categoria, prioridad)]

            # Una incidencia de cada combinación va a entrenamiento
            train_base.append(primera)

            # Repartimos la otra entre validación y test
            if (CATEGORIAS.index(categoria) + indice) % 2 == 0:
                validation_base.append(segunda)
            else:
                test_base.append(segunda)

    # Comprobamos el reparto de los 24 problemas originales
    assert len(train_base) == 12
    assert len(validation_base) == 6
    assert len(test_base) == 6


    # 4. GENERAR LOS EJEMPLOS SINTÉTICOS

    # Generamos las variaciones de cada conjunto por separado
    train = generar_ejemplos(train_base, N_TRAIN)
    validation = generar_ejemplos(validation_base, N_VALIDACION)
    test = generar_ejemplos(test_base, N_TEST)

    # Verificamos que cada conjunto tenga el tamaño esperado
    assert len(train) == N_TRAIN, "Cantidad incorrecta en train"
    assert len(validation) == N_VALIDACION, "Cantidad incorrecta en validation"
    assert len(test) == N_TEST, "Cantidad incorrecta en test"


    # 5. COMPROBAR DUPLICADOS Y DISTRIBUCIÓN

    # Reunimos todos los textos para comprobar que sean únicos
    textos = [ejemplo["incidencia"] for ejemplo in train + validation + test]

    assert len(textos) == len(set(textos)), "Hay incidencias duplicadas"

    # Verificamos que el test incluya las cuatro categorías
    assert set(e["categoria"] for e in test) == set(CATEGORIAS)

    # Verificamos que el test incluya las tres prioridades
    assert set(e["prioridad"] for e in test) == set(PRIORIDADES)


    # 6. GUARDAR LOS DATASETS

    guardar_jsonl("train.jsonl", train)
    guardar_jsonl("validation.jsonl", validation)
    guardar_jsonl("test.jsonl", test)

    print("\nDataset generado correctamente:")
    print(f"  Entrenamiento: {len(train)} ejemplos")
    print(f"  Validación:    {len(validation)} ejemplos")
    print(f"  Prueba:        {len(test)} ejemplos")
    print("  Duplicados:    0")

    # Mostramos la distribución de etiquetas del test
    print("\nDistribución del test:")

    for categoria in CATEGORIAS:
        cantidad = sum(e["categoria"] == categoria for e in test)
        print(f"  {categoria}: {cantidad}")

    for prioridad in PRIORIDADES:
        cantidad = sum(e["prioridad"] == prioridad for e in test)
        print(f"  {prioridad}: {cantidad}")

if __name__ == "__main__":
    preparar_datasets()
