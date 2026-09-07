import cv2
from ultralytics import YOLO

# --------------------------------------------------
# MODELO ENTRENADO
# --------------------------------------------------

modelo = YOLO(r"runs\detect\train-3\weights\best.pt")

# --------------------------------------------------
# VIDEO
# --------------------------------------------------

video = cv2.VideoCapture("prueba3b.mp4")

print("¿Video abierto?:", video.isOpened())


while True:

    ret, frame = video.read()

    if not ret:
        break

    alto, ancho = frame.shape[:2]

    # --------------------------------------------------
    # ZOOM HACIA LA IZQUIERDA
    # --------------------------------------------------

    x1_zoom = int(ancho * 0.00)
    x2_zoom = int(ancho * 0.55)

    y1_zoom = int(alto * 0.05)
    y2_zoom = int(alto * 0.90)

    zona_zoom = frame[
        y1_zoom:y2_zoom,
        x1_zoom:x2_zoom
    ]

    frame_zoom = cv2.resize(
        zona_zoom,
        (ancho, alto),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------
    # DETECCIÓN
    # --------------------------------------------------

    resultados = modelo(
        frame_zoom,
        conf=0.20,
        imgsz=1280,
        verbose=False
    )

    # Copia sobre la que vamos a dibujar nosotros
    frame_resultado = frame_zoom.copy()

    # --------------------------------------------------
    # LEER Y DIBUJAR DETECCIONES
    # --------------------------------------------------

    for box in resultados[0].boxes:

        clase = int(box.cls[0])
        confianza = float(box.conf[0])

        nombre = modelo.names[clase]

        bx1, by1, bx2, by2 = map(
            int,
            box.xyxy[0]
        )

        print(
            nombre,
            "| confianza:",
            round(confianza, 2),
            "| caja:",
            bx1, by1, bx2, by2
        )

        # Solo nos interesan casco / sin casco
        if nombre not in ["helmet", "no_helmet"]:
            continue

        # Dibujar bounding box
        cv2.rectangle(
            frame_resultado,
            (bx1, by1),
            (bx2, by2),
            (255, 255, 255),
            3
        )

        texto = f"{nombre} {confianza:.2f}"

        cv2.putText(
            frame_resultado,
            texto,
            (bx1, max(by1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # --------------------------------------------------
    # MOSTRAR
    # --------------------------------------------------

    cv2.imshow(
        "Deteccion casco DEBUG",
        frame_resultado
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()