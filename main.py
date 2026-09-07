import cv2
import json
from datetime import datetime
from ultralytics import YOLO


# --------------------------------------------------
# MODELOS
# --------------------------------------------------

# Detector general
modelo_general = YOLO("yolo11s.pt")

# Clasificador casco / no casco
modelo_clasificador = YOLO(
    r"weights\best.pt"
)


# --------------------------------------------------
# VIDEO
# --------------------------------------------------

video = cv2.VideoCapture("pruebab.mp4")

print("¿Video abierto?:", video.isOpened())


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

UMBRAL_CLASIFICACION = 0.75

# Número de frames consecutivos necesarios
# para confirmar una infracción
FRAMES_CONFIRMACION = 3


# --------------------------------------------------
# MEMORIA POR MOTO
# --------------------------------------------------

# Guarda cuántos frames consecutivos lleva
# cada moto clasificada como no_helmet
contador_no_helmet = {}

# IDs de motos que ya generaron una alerta
motos_reportadas = set()

# Guardamos aquí todas las infracciones
# para mostrarlas nuevamente al final
infracciones_detectadas = []


# --------------------------------------------------
# FUNCIONES
# --------------------------------------------------

def centro_caja(caja):

    x1, y1, x2, y2 = caja

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


def persona_cercana_a_moto(caja_moto, personas):

    """
    Busca la persona más cercana a una motocicleta.
    """

    mx, my = centro_caja(caja_moto)

    mejor_persona = None
    mejor_distancia = float("inf")

    for persona in personas:

        px, py = centro_caja(
            persona["caja"]
        )

        distancia = (
            (mx - px) ** 2 +
            (my - py) ** 2
        ) ** 0.5

        if distancia < mejor_distancia:

            mejor_distancia = distancia
            mejor_persona = persona

    return mejor_persona, mejor_distancia


# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    alto_frame, ancho_frame = frame.shape[:2]


    # --------------------------------------------------
    # DETECCIÓN + TRACKING
    # --------------------------------------------------

    resultados = modelo_general.track(
        frame,

        # 0 = person
        # 3 = motorcycle
        classes=[0, 3],

        conf=0.15,
        imgsz=960,

        # Mantener IDs entre frames
        persist=True,

        verbose=False
    )


    personas = []
    motos = []


    # --------------------------------------------------
    # LEER DETECCIONES
    # --------------------------------------------------

    for box in resultados[0].boxes:

        clase = int(box.cls[0])
        confianza = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        deteccion = {
            "confianza": confianza,
            "caja": (x1, y1, x2, y2)
        }


        # PERSONA
        if clase == 0:

            personas.append(
                deteccion
            )


        # MOTO
        elif clase == 3:

            if box.id is not None:

                moto_id = int(
                    box.id[0]
                )

            else:

                moto_id = -1


            deteccion["id"] = moto_id

            motos.append(
                deteccion
            )


    # --------------------------------------------------
    # FRAME LIMPIO
    # --------------------------------------------------

    # No mostramos las cajas de las personas
    frame_resultado = frame.copy()


    # --------------------------------------------------
    # ANALIZAR CADA MOTO
    # --------------------------------------------------

    for moto in motos:

        moto_id = moto["id"]

        x1_m, y1_m, x2_m, y2_m = moto["caja"]

        ancho_moto = x2_m - x1_m
        alto_moto = y2_m - y1_m


        # --------------------------------------------------
        # DIBUJAR SOLO LA MOTO
        # --------------------------------------------------

        cv2.rectangle(
            frame_resultado,
            (x1_m, y1_m),
            (x2_m, y2_m),
            (255, 255, 255),
            2
        )


        # --------------------------------------------------
        # BUSCAR PERSONA CERCA DE LA MOTO
        # --------------------------------------------------

        persona, distancia = persona_cercana_a_moto(
            moto["caja"],
            personas
        )


        # --------------------------------------------------
        # CASO 1:
        # HAY PERSONA CERCA
        # --------------------------------------------------

        if (
            persona is not None
            and distancia < ancho_moto * 2
        ):

            x1_p, y1_p, x2_p, y2_p = persona["caja"]

            alto_persona = y2_p - y1_p

            # Tomamos aproximadamente el 30%
            # superior de la persona
            cabeza_y2 = y1_p + int(
                alto_persona * 0.30
            )

            crop = frame[
                y1_p:cabeza_y2,
                x1_p:x2_p
            ]

            origen_crop = "persona"


        # --------------------------------------------------
        # CASO 2:
        # NO HAY PERSONA CONFIABLE
        # --------------------------------------------------

        else:

            # Estimamos una zona encima de la moto
            # donde debería estar el conductor

            x1_crop = max(
                0,
                x1_m - int(ancho_moto * 0.20)
            )

            x2_crop = min(
                ancho_frame,
                x2_m + int(ancho_moto * 0.20)
            )

            y1_crop = max(
                0,
                y1_m - int(alto_moto * 1.5)
            )

            y2_crop = min(
                alto_frame,
                y1_m + int(alto_moto * 0.20)
            )

            crop = frame[
                y1_crop:y2_crop,
                x1_crop:x2_crop
            ]

            origen_crop = "estimado"


        # --------------------------------------------------
        # VALIDAR CROP
        # --------------------------------------------------

        if crop.size == 0:
            continue


        # --------------------------------------------------
        # AGRANDAR PARA EL CLASIFICADOR
        # --------------------------------------------------

        crop_224 = cv2.resize(
            crop,
            (224, 224),
            interpolation=cv2.INTER_CUBIC
        )


        # --------------------------------------------------
        # MOSTRAR QUÉ ESTÁ VIENDO EL CLASIFICADOR
        # --------------------------------------------------

        cv2.imshow(
            "Crop clasificador",
            crop_224
        )


        # --------------------------------------------------
        # CLASIFICAR CASCO
        # --------------------------------------------------

        resultado_clasificacion = modelo_clasificador(
            crop_224,
            imgsz=224,
            verbose=False
        )

        probs = resultado_clasificacion[0].probs

        clase = probs.top1

        confianza = float(
            probs.top1conf
        )

        nombre = modelo_clasificador.names[
            clase
        ]


        # --------------------------------------------------
        # DECIDIR RESULTADO
        # --------------------------------------------------

        if confianza < UMBRAL_CLASIFICACION:

            estado = "INCIERTO"

        elif nombre == "helmet":

            estado = "CASCO"

        elif nombre == "no_helmet":

            estado = "SIN CASCO"

        else:

            estado = "INCIERTO"


        # --------------------------------------------------
        # CONTADOR POR MOTO
        # --------------------------------------------------

        if (
            nombre == "no_helmet"
            and confianza >= UMBRAL_CLASIFICACION
        ):

            contador_no_helmet[moto_id] = (
                contador_no_helmet.get(
                    moto_id,
                    0
                )
                + 1
            )

        else:

            contador_no_helmet[moto_id] = 0


        frames_sin_casco = contador_no_helmet.get(
            moto_id,
            0
        )


        # --------------------------------------------------
        # MOSTRAR INFO EN TERMINAL
        # --------------------------------------------------

        print(
            "Moto ID:",
            moto_id,
            "| Crop:", origen_crop,
            "| Resultado:", nombre,
            "| Confianza:", round(confianza, 2),
            "| Frames sin casco:",
            frames_sin_casco
        )


        # --------------------------------------------------
        # TEXTO EN PANTALLA
        # --------------------------------------------------

        cv2.putText(
            frame_resultado,
            f"MOTO ID {moto_id}",
            (x1_m, max(y1_m - 25, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame_resultado,
            f"{estado} {confianza:.2f}",
            (x1_m, max(y1_m - 5, 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # --------------------------------------------------
        # CONFIRMAR INFRACCIÓN
        # --------------------------------------------------

        if (
            frames_sin_casco >= FRAMES_CONFIRMACION
            and moto_id not in motos_reportadas
        ):

            motos_reportadas.add(
                moto_id
            )


            # --------------------------------------------------
            # EVENTO PARA EL LLM
            # --------------------------------------------------

            evento = {

                "vehiculo": "moto",

                "camara": "Cam_A",

                "hora": datetime.now().strftime(
                    "%H:%M:%S"
                ),

                "patente": None,

                "infraccion": "sin_casco",

                "velocidad_kmh": None

            }


            # Guardar infracción
            infracciones_detectadas.append(
                evento
            )


            # Mostrar inmediatamente también
            print("\n")
            print(
                "================================="
            )

            print(
                "INFRACCION CONFIRMADA"
            )

            print(
                "Moto ID:",
                moto_id
            )

            print(
                json.dumps(
                    evento,
                    indent=4,
                    ensure_ascii=False
                )
            )

            print(
                "================================="
            )


        # --------------------------------------------------
        # MOSTRAR ALERTA
        # --------------------------------------------------

        if moto_id in motos_reportadas:

            cv2.putText(
                frame_resultado,
                "SIN CASCO CONFIRMADO",
                (x1_m, y2_m + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # --------------------------------------------------
    # MOSTRAR VIDEO PRINCIPAL
    # --------------------------------------------------

    cv2.imshow(
        "Deteccion de casco",
        frame_resultado
    )


    # Q para cerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# RESUMEN FINAL
# --------------------------------------------------

print("\n")
print(
    "================================="
)

print(
    "RESUMEN FINAL DE INFRACCIONES"
)

print(
    "================================="
)


if len(infracciones_detectadas) == 0:

    print(
        "No se detectaron infracciones."
    )

else:

    for i, evento in enumerate(
        infracciones_detectadas,
        start=1
    ):

        print(
            f"\nInfraccion {i}:"
        )

        print(
            json.dumps(
                evento,
                indent=4,
                ensure_ascii=False
            )
        )


print(
    "\nTotal de infracciones:",
    len(infracciones_detectadas)
)

print(
    "================================="
)


# --------------------------------------------------
# CERRAR
# --------------------------------------------------

video.release()

cv2.destroyAllWindows()