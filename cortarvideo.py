import cv2

entrada = "videomotos.mp4"
salida = "videomotosb.mp4"

inicio_seg = 0
fin_seg = 30

video = cv2.VideoCapture(entrada)

fps = video.get(cv2.CAP_PROP_FPS)
ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

inicio_frame = int(inicio_seg * fps)
fin_frame = int(fin_seg * fps)

video.set(cv2.CAP_PROP_POS_FRAMES, inicio_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    salida,
    fourcc,
    fps,
    (ancho, alto)
)

frame_actual = inicio_frame

while frame_actual < fin_frame:
    ret, frame = video.read()

    if not ret:
        break

    writer.write(frame)
    frame_actual += 1

video.release()
writer.release()

print("Video recortado guardado como:", salida)