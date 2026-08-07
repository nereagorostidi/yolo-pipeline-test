from ultralytics import YOLO

# Cargar la arquitectura ligera nano de YOLO11 (version mas reciente que YOLOv8)
model = YOLO('yolo11n.pt')

# Ejecutar entrenamiento local
results = model.train(
    data='data.yaml',
    epochs=80,      # tope de pasadas completas al dataset; no tiene por que llegar a las 80
    patience=20,    # si en 20 epocas seguidas no mejora en validacion, para solo (evita overfitting)
    imgsz=640,
    batch=8,
    name='mi_modelo_frutas'
)

print('¡Entrenamiento completado! El archivo resultante se ha guardado en:')
print('runs/detect/mi_modelo_frutas/weights/best.pt')