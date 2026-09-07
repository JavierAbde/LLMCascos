import os
import cv2
import yaml

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

ORIGEN = "CASCO"
DESTINO = "CASCO_CLASIFICACION"

DATA_YAML = os.path.join(ORIGEN, "data.yaml")


# --------------------------------------------------
# LEER NOMBRES DE CLASES
# --------------------------------------------------

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, list):
    mapa = {i: nombre for i, nombre in enumerate(names)}
else:
    mapa = {int(k): v for k, v in names.items()}

print("Clases encontradas:")
for k, v in mapa.items():
    print(k, "=", v)


# --------------------------------------------------
# ENCONTRAR IDS DE HELMET Y NO_HELMET
# --------------------------------------------------

id_helmet = None
id_no_helmet = None

for k, nombre in mapa.items():
    nombre_normalizado = nombre.lower().strip()

    if nombre_normalizado == "helmet":
        id_helmet = k

    elif nombre_normalizado in [
        "no_helmet",
        "no helmet",
        "no-helmet"
    ]:
        id_no_helmet = k


if id_helmet is None or id_no_helmet is None:
    print("ERROR: no encontré helmet o no_helmet")
    exit()


print("Helmet:", id_helmet)
print("No helmet:", id_no_helmet)


# --------------------------------------------------
# CREAR CARPETAS
# --------------------------------------------------

for split in ["train", "valid", "test"]:

    os.makedirs(
        os.path.join(
            DESTINO,
            split,
            "helmet"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            DESTINO,
            split,
            "no_helmet"
        ),
        exist_ok=True
    )


# --------------------------------------------------
# PROCESAR DATASET
# --------------------------------------------------

contador_helmet = 0
contador_nohelmet = 0

for split in ["train", "valid", "test"]:

    carpeta_imagenes = os.path.join(
        ORIGEN,
        split,
        "images"
    )

    carpeta_labels = os.path.join(
        ORIGEN,
        split,
        "labels"
    )

    if not os.path.exists(carpeta_labels):
        continue

    for archivo_label in os.listdir(carpeta_labels):

        if not archivo_label.endswith(".txt"):
            continue

        nombre_base = os.path.splitext(
            archivo_label
        )[0]

        # Buscar imagen correspondiente
        ruta_imagen = None

        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp"
        ]:

            candidato = os.path.join(
                carpeta_imagenes,
                nombre_base + ext
            )

            if os.path.exists(candidato):
                ruta_imagen = candidato
                break

        if ruta_imagen is None:
            continue

        imagen = cv2.imread(ruta_imagen)

        if imagen is None:
            continue

        alto, ancho = imagen.shape[:2]

        ruta_label = os.path.join(
            carpeta_labels,
            archivo_label
        )

        with open(
            ruta_label,
            "r",
            encoding="utf-8"
        ) as f:

            lineas = f.readlines()

        for i, linea in enumerate(lineas):

            partes = linea.strip().split()

            if len(partes) < 5:
                continue

            clase = int(partes[0])

            # Ignorar license_plate y cualquier otra clase
            if clase not in [
                id_helmet,
                id_no_helmet
            ]:
                continue

            x_centro = float(partes[1])
            y_centro = float(partes[2])
            w = float(partes[3])
            h = float(partes[4])

            # Convertir coordenadas YOLO normalizadas a píxeles
            x1 = int(
                (x_centro - w / 2) * ancho
            )

            y1 = int(
                (y_centro - h / 2) * alto
            )

            x2 = int(
                (x_centro + w / 2) * ancho
            )

            y2 = int(
                (y_centro + h / 2) * alto
            )

            # Agregar un pequeño margen alrededor
            margen_x = int((x2 - x1) * 0.20)
            margen_y = int((y2 - y1) * 0.20)

            x1 = max(0, x1 - margen_x)
            y1 = max(0, y1 - margen_y)

            x2 = min(ancho, x2 + margen_x)
            y2 = min(alto, y2 + margen_y)

            crop = imagen[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # --------------------------------------------------
            # GUARDAR SEGÚN CLASE
            # --------------------------------------------------

            if clase == id_helmet:

                nombre_salida = (
                    f"helmet_{contador_helmet}.jpg"
                )

                ruta_salida = os.path.join(
                    DESTINO,
                    split,
                    "helmet",
                    nombre_salida
                )

                contador_helmet += 1

            else:

                nombre_salida = (
                    f"nohelmet_{contador_nohelmet}.jpg"
                )

                ruta_salida = os.path.join(
                    DESTINO,
                    split,
                    "no_helmet",
                    nombre_salida
                )

                contador_nohelmet += 1

            cv2.imwrite(
                ruta_salida,
                crop
            )


# --------------------------------------------------
# RESULTADO
# --------------------------------------------------

print("\nDATASET DE CLASIFICACIÓN CREADO")
print("Helmet:", contador_helmet)
print("No helmet:", contador_nohelmet)
print("Carpeta:", DESTINO)