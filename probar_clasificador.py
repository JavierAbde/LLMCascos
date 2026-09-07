from ultralytics import YOLO
import cv2

# Clasificador entrenado
modelo = YOLO(r"runs\classify\train\weights\best.pt")

# Pon aquí una imagen de casco del dataset de validación
imagen = r"CASCO_CLASIFICACION\valid\no_helmet\nohelmet_725.jpg"

resultado = modelo(
    imagen,
    imgsz=224,
    verbose=False
)

probs = resultado[0].probs

clase = probs.top1
confianza = float(probs.top1conf)

nombre = modelo.names[clase]

print("Resultado:", nombre)
print("Confianza:", round(confianza, 3))

# Mostrar la imagen que se clasificó
img = cv2.imread(imagen)

cv2.putText(
    img,
    f"{nombre} {confianza:.2f}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.imshow("Prueba clasificador", img)
cv2.waitKey(0)
cv2.destroyAllWindows()