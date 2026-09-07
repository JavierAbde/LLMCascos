import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# MODELO
# --------------------------------------------------

modelo = YOLO("yolo11s.pt")

# --------------------------------------------------
# VIDEO
# --------------------------------------------------

video = cv2.VideoCapture("videobici.mp4")

print("¿Video abierto?:", video.isOpened())

# --------------------------------------------------
# VARIABLES PARA POLÍGONO
# --------------------------------------------------

puntos_poligono = []
poligono_listo = False

# --------------------------------------------------
# MOUSE
# --------------------------------------------------

def click_mouse(event, x, y, flags, param):
    global puntos_poligono, poligono_listo

    # Click izquierdo = agregar punto
    if event == cv2.EVENT_LBUTTONDOWN and not poligono_listo:
        puntos_poligono.append((x, y))
        print("Punto agregado:", (x, y))

    # Click derecho = cerrar polígono
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(puntos_poligono) >= 3:
            poligono_listo = True
            print("POLÍGONO CERRADO")


# --------------------------------------------------
# LEER PRIMER FRAME
# --------------------------------------------------

ret, primer_frame = video.read()

if not ret:
    print("No se pudo leer el video.")
    video.release()
    exit()

# --------------------------------------------------
# DEFINIR CALLE MANUALMENTE
# --------------------------------------------------

cv2.namedWindow("Definir calle")
cv2.setMouseCallback("Definir calle", click_mouse)

print("")
print("INSTRUCCIONES:")
print("- Click IZQUIERDO: agregar puntos de la calle")
print("- Click DERECHO: cerrar polígono")
print("- Presiona ENTER cuando termines")
print("")

while True:

    frame_dibujo = primer_frame.copy()

    # Dibujar puntos
    for punto in puntos_poligono:
        cv2.circle(
            frame_dibujo,
            punto,
            5,
            (255, 255, 255),
            -1
        )

    # Dibujar líneas entre puntos
    if len(puntos_poligono) > 1:

        for i in range(len(puntos_poligono) - 1):
            cv2.line(
                frame_dibujo,
                puntos_poligono[i],
                puntos_poligono[i + 1],
                (255, 255, 255),
                2
            )

    # Cerrar visualmente el polígono
    if poligono_listo and len(puntos_poligono) >= 3:

        pts = np.array(
            puntos_poligono,
            np.int32
        )

        cv2.polylines(
            frame_dibujo,
            [pts],
            True,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Definir calle",
        frame_dibujo
    )

    tecla = cv2.waitKey(20) & 0xFF

    # ENTER
    if tecla == 13 and poligono_listo:
        break

    # Q para cancelar
    if tecla == ord("q"):
        video.release()
        cv2.destroyAllWindows()
        exit()


cv2.destroyWindow("Definir calle")

# --------------------------------------------------
# VOLVER AL INICIO DEL VIDEO
# --------------------------------------------------

video.set(
    cv2.CAP_PROP_POS_FRAMES,
    0
)

poligono = np.array(
    puntos_poligono,
    dtype=np.int32
)

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    # --------------------------------------------------
    # DETECTAR BICICLETAS
    # --------------------------------------------------

    resultados = modelo(
        frame,
        classes=[1],   # 1 = bicycle
        conf=0.20,
        imgsz=960,
        verbose=False
    )

    frame_resultado = frame.copy()

    # --------------------------------------------------
    # DIBUJAR CALLE / ZONA PERMITIDA
    # --------------------------------------------------

    overlay = frame_resultado.copy()

    cv2.fillPoly(
        overlay,
        [poligono],
        (255, 255, 255)
    )

    frame_resultado = cv2.addWeighted(
        overlay,
        0.15,
        frame_resultado,
        0.85,
        0
    )

    cv2.polylines(
        frame_resultado,
        [poligono],
        True,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_resultado,
        "CALLE / ZONA PERMITIDA",
        tuple(poligono[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------
    # ANALIZAR BICICLETAS
    # --------------------------------------------------

    for box in resultados[0].boxes:

        confianza = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        # Centro de la bicicleta
        centro_x = int(
            (x1 + x2) / 2
        )

        centro_y = int(
            (y1 + y2) / 2
        )

        centro = (
            centro_x,
            centro_y
        )

        # --------------------------------------------------
        # VER SI ESTÁ DENTRO DE LA CALLE
        # --------------------------------------------------

        dentro = cv2.pointPolygonTest(
            poligono,
            centro,
            False
        )

        # DENTRO = calle / permitido
        if dentro >= 0:
            estado = "BICICLETA EN CALLE"

        # FUERA = vereda / infracción
        else:
            estado = "BICICLETA EN VEREDA"

        # --------------------------------------------------
        # DIBUJAR BICICLETA
        # --------------------------------------------------

        cv2.rectangle(
            frame_resultado,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.circle(
            frame_resultado,
            centro,
            5,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            frame_resultado,
            f"{estado} {confianza:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # --------------------------------------------------
        # ALERTA SOLO SI ESTÁ FUERA DE LA CALLE
        # --------------------------------------------------

        if dentro < 0:

            print(
                "ALERTA:",
                estado,
                "| Confianza:",
                round(confianza, 2)
            )

    # --------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------

    cv2.imshow(
        "Bicicletas - Calle y vereda",
        frame_resultado
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()