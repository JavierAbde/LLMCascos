import cv2
from ultralytics import YOLO

# --------------------------------------------------
# CARGAR MODELO DE CASCO
# --------------------------------------------------

modelo = YOLO("helmet_best.pt")

print("SCRIPT INICIADO")

# --------------------------------------------------
# ABRIR VIDEO
# --------------------------------------------------

video = cv2.VideoCapture("prueba2.mp4")

print("¿Video abierto?:", video.isOpened())


# --------------------------------------------------
# PROCESAR VIDEO
# --------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    # --------------------------------------------------
    # DETECCIÓN CON MODELO DE CASCO
    # --------------------------------------------------

    resultados = modelo(
        frame,
        conf=0.35,
        imgsz=960,
        verbose=False
    )

    # --------------------------------------------------
    # MOSTRAR DETECCIONES EN TERMINAL
    # --------------------------------------------------

    for box in resultados[0].boxes:

        clase = int(box.cls[0])
        confianza = float(box.conf[0])

        nombre = modelo.names[clase]

        print(
            nombre,
            "| confianza:",
            round(confianza, 2)
        )

    # --------------------------------------------------
    # DIBUJAR CAJAS
    # --------------------------------------------------

    frame_resultado = resultados[0].plot()

    # --------------------------------------------------
    # MOSTRAR VIDEO
    # --------------------------------------------------

    cv2.imshow(
        "Prueba modelo casco",
        frame_resultado
    )

    # Presiona Q para cerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# CERRAR
# --------------------------------------------------

video.release()
cv2.destroyAllWindows()

print("VIDEO TERMINADO")