import argparse
import os
import cv2
from ultralytics import YOLO

# Cargar VUESTRO cerebro entrenado
model = YOLO('runs/detect/mi_modelo_frutas-80/weights/best.pt')

# Recibir el nombre del video y los parametros de deteccion por linea de comandos
parser = argparse.ArgumentParser(
    description='Detector de frutas YOLO sobre un video.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('video_path', help='Ruta del video a analizar (ej. video_dron.mp4)')
parser.add_argument('--conf', type=float, default=0.5,
                     help='Confianza minima para mostrar una deteccion (subir = menos falsos positivos, bajar = menos frutas sin detectar)')
parser.add_argument('--vid-stride', type=int, default=3,
                     help='Analiza 1 de cada N frames (1 = analiza todos; subirlo va mas rapido pero puede saltarse frutas que pasan rapido)')
parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True,
                     help='Test-time augmentation: analiza cada frame varias veces (flips/escalas) y combina resultados, mas preciso pero mas lento. Usa --no-augment para desactivarlo')
args = parser.parse_args()

video_path = args.video_path
VID_STRIDE = args.vid_stride
if not os.path.isfile(video_path):
    raise SystemExit(f'No encuentro "{video_path}"')

# fps original del video, para que el output.mp4 dure lo mismo que el original
cap_info = cv2.VideoCapture(video_path)
fps_original = cap_info.get(cv2.CAP_PROP_FPS)
cap_info.release()

# stream=True: procesa el video frame a frame (con el salto de vid_stride)
# sin cargarlo entero en memoria
results = model.predict(
    source=video_path,
    imgsz=640,       # igual que el imgsz de entrenamiento; a menos resolucion se pierde detalle y confunde mas las clases
    conf=args.conf,        # confianza minima para mostrar una deteccion (subir = menos falsos positivos, bajar = menos frutas sin detectar)
    vid_stride=VID_STRIDE,    # 1 = analiza todos los frames; subirlo va mas rapido pero puede saltarse frutas que pasan rapido
    stream=True,
    verbose=False,
    augment=args.augment     # test-time augmentation: analiza cada frame varias veces (flips/escalas) y combina resultados, mas preciso pero mas lento
)

writer = None
for r in results:
    annotated_frame = r.plot()

    if writer is None:
        h, w = annotated_frame.shape[:2]
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', os.path.basename(video_path))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_original / VID_STRIDE,
            (w, h),
        )
    writer.write(annotated_frame)

    # Redimensionar para la vista previa manteniendo la proporcion original
    # (960x540 fijo deformaba los videos verticales del iPhone)
    h, w = annotated_frame.shape[:2]
    escala = 540 / h
    preview = cv2.resize(annotated_frame, (round(w * escala), 540))

    # Mostrar ventana interactiva
    cv2.imshow('Detector de Frutas YOLO', preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cv2.destroyAllWindows()
