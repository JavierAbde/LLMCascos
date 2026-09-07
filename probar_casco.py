from ultralytics import YOLO

modelo_casco = YOLO("helmet_best.pt")

print("Modelo cargado correctamente")
print("Clases del modelo:")
print(modelo_casco.names)