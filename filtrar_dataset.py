import os
import shutil
import yaml

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ORIGEN = "CASCO"
DESTINO = "CASCO_2CLASES"

DATA_YAML = os.path.join(ORIGEN, "data.yaml")


# --------------------------------------------------
# LEER CLASES DEL DATASET ORIGINAL
# --------------------------------------------------

with open(DATA_YAML, "r", encoding="utf-8") as archivo:
    data = yaml.safe_load(archivo)

names = data["names"]

# Roboflow puede guardar "names" como lista o diccionario
if isinstance(names, list):
    mapa_nombres = {
        i: nombre
        for i, nombre in enumerate(names)
    }
else:
    mapa_nombres = {
        int(i): nombre
        for i, nombre in names.items()
    }


print("Clases originales:")

for clase_id, nombre in mapa_nombres.items():
    print(clase_id, "=", nombre)


# --------------------------------------------------
# IDENTIFICAR CLASES QUE QUEREMOS CONSERVAR
# --------------------------------------------------

id_helmet = None
id_no_helmet = None

for clase_id, nombre in mapa_nombres.items():

    nombre_normalizado = nombre.lower().strip()

    if nombre_normalizado == "helmet":
        id_helmet = clase_id

    elif nombre_normalizado in [
        "no_helmet",
        "no helmet",
        "no-helmet"
    ]:
        id_no_helmet = clase_id


if id_helmet is None or id_no_helmet is None:

    print("\nERROR:")
    print("No pude encontrar helmet y no_helmet en data.yaml.")
    exit()


print("\nClases seleccionadas:")
print("Helmet original:", id_helmet)
print("No Helmet original:", id_no_helmet)


# --------------------------------------------------
# NUEVA NUMERACIÓN
#
# 0 = helmet
# 1 = no_helmet
# --------------------------------------------------

remapeo = {
    id_helmet: 0,
    id_no_helmet: 1
}


# --------------------------------------------------
# CREAR DATASET NUEVO
# --------------------------------------------------

if os.path.exists(DESTINO):
    shutil.rmtree(DESTINO)

os.makedirs(DESTINO)


divisiones = [
    "train",
    "valid",
    "test"
]


for division in divisiones:

    imagenes_origen = os.path.join(
        ORIGEN,
        division,
        "images"
    )

    etiquetas_origen = os.path.join(
        ORIGEN,
        division,
        "labels"
    )

    imagenes_destino = os.path.join(
        DESTINO,
        division,
        "images"
    )

    etiquetas_destino = os.path.join(
        DESTINO,
        division,
        "labels"
    )

    os.makedirs(
        imagenes_destino,
        exist_ok=True
    )

    os.makedirs(
        etiquetas_destino,
        exist_ok=True
    )

    if not os.path.exists(etiquetas_origen):
        print(
            f"No existe {etiquetas_origen}, saltando..."
        )
        continue


    contador_imagenes = 0
    contador_etiquetas = 0


    # --------------------------------------------------
    # PROCESAR CADA ARCHIVO DE ETIQUETAS
    # --------------------------------------------------

    for archivo_label in os.listdir(etiquetas_origen):

        if not archivo_label.endswith(".txt"):
            continue

        ruta_label = os.path.join(
            etiquetas_origen,
            archivo_label
        )

        nuevas_lineas = []


        with open(
            ruta_label,
            "r",
            encoding="utf-8"
        ) as archivo:

            for linea in archivo:

                partes = linea.strip().split()

                if len(partes) < 5:
                    continue

                clase_original = int(partes[0])

                # Solo conservar helmet y no_helmet
                if clase_original not in remapeo:
                    continue

                nueva_clase = remapeo[
                    clase_original
                ]

                partes[0] = str(nueva_clase)

                nuevas_lineas.append(
                    " ".join(partes)
                )


        # --------------------------------------------------
        # SOLO COPIAR IMÁGENES QUE TENGAN
        # HELMET O NO_HELMET
        # --------------------------------------------------

        if len(nuevas_lineas) == 0:
            continue


        nombre_base = os.path.splitext(
            archivo_label
        )[0]


        # Buscar extensión de imagen
        imagen_encontrada = None

        for extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:

            candidato = os.path.join(
                imagenes_origen,
                nombre_base + extension
            )

            if os.path.exists(candidato):
                imagen_encontrada = candidato
                break


        if imagen_encontrada is None:
            print(
                "No encontré imagen para:",
                archivo_label
            )
            continue


        # Copiar imagen
        shutil.copy2(
            imagen_encontrada,
            imagenes_destino
        )


        # Guardar nueva etiqueta
        ruta_nueva_label = os.path.join(
            etiquetas_destino,
            archivo_label
        )

        with open(
            ruta_nueva_label,
            "w",
            encoding="utf-8"
        ) as archivo:

            archivo.write(
                "\n".join(nuevas_lineas)
            )


        contador_imagenes += 1
        contador_etiquetas += len(
            nuevas_lineas
        )


    print(
        f"{division}:",
        contador_imagenes,
        "imágenes |",
        contador_etiquetas,
        "anotaciones"
    )


# --------------------------------------------------
# CREAR NUEVO data.yaml
# --------------------------------------------------

nuevo_yaml = {
    "path": os.path.abspath(DESTINO),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",

    "nc": 2,

    "names": [
        "helmet",
        "no_helmet"
    ]
}


ruta_nuevo_yaml = os.path.join(
    DESTINO,
    "data.yaml"
)


with open(
    ruta_nuevo_yaml,
    "w",
    encoding="utf-8"
) as archivo:

    yaml.safe_dump(
        nuevo_yaml,
        archivo,
        sort_keys=False,
        allow_unicode=True
    )


print("\n--------------------------------")
print("DATASET NUEVO CREADO")
print("--------------------------------")

print(
    "Carpeta:",
    DESTINO
)

print(
    "Configuración:",
    ruta_nuevo_yaml
)

print("\nClases nuevas:")
print("0 = helmet")
print("1 = no_helmet")