import cv2
from ultralytics import YOLO

# --------------------------------------------------
# MODELO
# --------------------------------------------------

modelo = YOLO("yolo11s.pt")

# --------------------------------------------------
# VIDEO
# --------------------------------------------------

video = cv2.VideoCapture("prueba3b.mp4")

print("¿Video abierto?:", video.isOpened())

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

# Distancia REAL entre las dos líneas.
# Para la PoC puedes usar, por ejemplo, 20 metros.
DISTANCIA_METROS = 20.0

# Límite de velocidad
LIMITE_KMH = 50

# Puntos elegidos con el mouse
puntos = []

lineas_listas = False

# Guarda el frame donde cada vehículo cruza la línea 1
cruces_linea_1 = {}

# Guarda velocidades ya calculadas
velocidades = {}

# Posición anterior de cada vehículo
posiciones_anteriores = {}

# --------------------------------------------------
# FPS DEL VIDEO
# --------------------------------------------------

fps = video.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)

# --------------------------------------------------
# FUNCIÓN DEL MOUSE
# --------------------------------------------------

def click_mouse(event, x, y, flags, param):
    global puntos, lineas_listas

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(puntos) < 4:
            puntos.append((x, y))
            print("Punto agregado:", (x, y))

        if len(puntos) == 4:
            lineas_listas = True
            print("LÍNEAS DEFINIDAS")


# --------------------------------------------------
# LEER PRIMER FRAME
# --------------------------------------------------

ret, primer_frame = video.read()

if not ret:
    print("No se pudo leer el video.")
    video.release()
    exit()

# --------------------------------------------------
# DEFINIR LÍNEAS
# --------------------------------------------------

cv2.namedWindow("Definir lineas")
cv2.setMouseCallback("Definir lineas", click_mouse)

print("")
print("INSTRUCCIONES:")
print("- Click 1 y 2: extremos de LINEA 1")
print("- Click 3 y 4: extremos de LINEA 2")
print("- Presiona ENTER cuando termines")
print("")

while True:

    frame_dibujo = primer_frame.copy()

    # Dibujar puntos
    for punto in puntos:
        cv2.circle(
            frame_dibujo,
            punto,
            6,
            (255, 255, 255),
            -1
        )

    # Dibujar línea 1
    if len(puntos) >= 2:
        cv2.line(
            frame_dibujo,
            puntos[0],
            puntos[1],
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame_dibujo,
            "LINEA 1",
            puntos[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Dibujar línea 2
    if len(puntos) >= 4:
        cv2.line(
            frame_dibujo,
            puntos[2],
            puntos[3],
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame_dibujo,
            "LINEA 2",
            puntos[2],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Definir lineas",
        frame_dibujo
    )

    tecla = cv2.waitKey(20) & 0xFF

    if tecla == 13 and lineas_listas:
        break

    if tecla == ord("q"):
        video.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Definir lineas")

# --------------------------------------------------
# GUARDAR COORDENADAS DE LAS LÍNEAS
# --------------------------------------------------

p1_l1, p2_l1 = puntos[0], puntos[1]
p1_l2, p2_l2 = puntos[2], puntos[3]

# Para este video vamos a usar la coordenada Y media
# de cada línea para detectar el cruce.
linea_1_y = int((p1_l1[1] + p2_l1[1]) / 2)
linea_2_y = int((p1_l2[1] + p2_l2[1]) / 2)

print("Línea 1 Y:", linea_1_y)
print("Línea 2 Y:", linea_2_y)

# --------------------------------------------------
# VOLVER AL INICIO DEL VIDEO
# --------------------------------------------------

video.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

frame_actual = 0

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_actual += 1

    # --------------------------------------------------
    # TRACKING
    # --------------------------------------------------

    resultados = modelo.track(
        frame,
        classes=[2, 3, 5, 7],
        # 2 = car
        # 3 = motorcycle
        # 5 = bus
        # 7 = truck
        conf=0.25,
        imgsz=960,
        persist=True,
        verbose=False
    )

    frame_resultado = frame.copy()

    # --------------------------------------------------
    # DIBUJAR LÍNEAS
    # --------------------------------------------------

    cv2.line(
        frame_resultado,
        p1_l1,
        p2_l1,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame_resultado,
        "LINEA 1",
        p1_l1,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.line(
        frame_resultado,
        p1_l2,
        p2_l2,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame_resultado,
        "LINEA 2",
        p1_l2,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # PROCESAR DETECCIONES
    # --------------------------------------------------

    if (
        resultados[0].boxes is not None
        and resultados[0].boxes.id is not None
    ):

        for box in resultados[0].boxes:

            vehiculo_id = int(box.id[0])

            clase = int(box.cls[0])
            confianza = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Usamos centro inferior del vehículo
            centro_x = int((x1 + x2) / 2)
            centro_y = y2

            posicion_actual = centro_y

            posicion_anterior = posiciones_anteriores.get(
                vehiculo_id,
                posicion_actual
            )

            # --------------------------------------------------
            # CRUCE DE LÍNEA 1
            # --------------------------------------------------

            cruzo_linea_1 = (
                posicion_anterior < linea_1_y
                and posicion_actual >= linea_1_y
            )

            if (
                cruzo_linea_1
                and vehiculo_id not in cruces_linea_1
            ):

                cruces_linea_1[
                    vehiculo_id
                ] = frame_actual

                print(
                    "Vehículo",
                    vehiculo_id,
                    "cruzó línea 1"
                )

            # --------------------------------------------------
            # CRUCE DE LÍNEA 2
            # --------------------------------------------------

            cruzo_linea_2 = (
                posicion_anterior < linea_2_y
                and posicion_actual >= linea_2_y
            )

            if (
                cruzo_linea_2
                and vehiculo_id in cruces_linea_1
                and vehiculo_id not in velocidades
            ):

                frame_inicio = cruces_linea_1[
                    vehiculo_id
                ]

                frames_transcurridos = (
                    frame_actual - frame_inicio
                )

                tiempo_segundos = (
                    frames_transcurridos / fps
                )

                if tiempo_segundos > 0:

                    velocidad_ms = (
                        DISTANCIA_METROS
                        / tiempo_segundos
                    )

                    velocidad_kmh = (
                        velocidad_ms * 3.6
                    )

                    velocidades[
                        vehiculo_id
                    ] = velocidad_kmh

                    print(
                        "Vehículo:",
                        vehiculo_id,
                        "| Tiempo:",
                        round(
                            tiempo_segundos,
                            2
                        ),
                        "s",
                        "| Velocidad:",
                        round(
                            velocidad_kmh,
                            1
                        ),
                        "km/h"
                    )

            # --------------------------------------------------
            # ACTUALIZAR POSICIÓN
            # --------------------------------------------------

            posiciones_anteriores[
                vehiculo_id
            ] = posicion_actual

            # --------------------------------------------------
            # DIBUJAR VEHÍCULO
            # --------------------------------------------------

            velocidad = velocidades.get(
                vehiculo_id
            )

            if velocidad is None:

                texto = (
                    f"ID {vehiculo_id}"
                )

            else:

                texto = (
                    f"ID {vehiculo_id} "
                    f"{velocidad:.1f} km/h"
                )

            cv2.rectangle(
                frame_resultado,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.circle(
                frame_resultado,
                (centro_x, centro_y),
                5,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame_resultado,
                texto,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # --------------------------------------------------
            # ALERTA
            # --------------------------------------------------

            if (
                velocidad is not None
                and velocidad > LIMITE_KMH
            ):

                cv2.putText(
                    frame_resultado,
                    "EXCESO DE VELOCIDAD",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # --------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------

    cv2.imshow(
        "Control de velocidad",
        frame_resultado
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()