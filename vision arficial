# Librerías necesarias
import cv2  # Para procesamiento de imágenes
import numpy as np  # Para operaciones con arrays
import matplotlib.pyplot as plt  # Para visualización de imágenes
import tensorflow as tf  # Para detección de objetos con redes neuronales

# Configuración para la captura de imágenes
camera = cv2.VideoCapture(0)  # Usa el índice adecuado si tienes varias cámaras

def capture_image():
    """Captura una imagen desde la cámara."""
    ret, frame = camera.read()
    if not ret:
        raise ValueError("No se pudo capturar la imagen de la cámara.")
    return frame

def process_image(image):
    """Procesa la imagen usando OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aplicar desenfoque gaussiano
    edges = cv2.Canny(blurred, 50, 150)  # Detectar bordes usando Canny
    return edges

def display_image(image, title="Imagen"):
    """Muestra la imagen usando matplotlib."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para visualización
    plt.title(title)
    plt.axis('off')  # No mostrar ejes
    plt.show()

def detect_objects(image):
    """Detecta objetos usando un modelo de TensorFlow (cargar un modelo previamente entrenado)."""
    model = tf.keras.models.load_model('cable_detector_model.h5')  # Cargar el modelo entrenado
    image_resized = cv2.resize(image, (150, 150))  # Redimensionar imagen
    image_array = np.expand_dims(image_resized, axis=0) / 255.0  # Normalizar
    prediction = model.predict(image_array)  # Realizar la predicción
    return prediction

# Capturar una imagen
image = capture_image()

# Procesar la imagen
processed_image = process_image(image)

# Mostrar la imagen procesada
display_image(processed_image, title="Imagen Procesada")

# Detectar objetos
prediction = detect_objects(image)
label = 'Cable Muerto' if prediction[0] > 0.5 else 'Cable OK'
print(f"Predicción: {label}")

# Mostrar la imagen con la etiqueta
cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if prediction[0] > 0.5 else (0, 0, 255), 2)
display_image(image, title="Imagen con Detección")

# Liberar la cámara
camera.release()
cv2.destroyAllWindows()


  import cv2

# Configuración de la Cámara
def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    # Inicialización de la Cámara
    camera = cv2.VideoCapture(camera_index)
    
    # Verificación de la apertura de la cámara
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    # Ajustes de la Cámara
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def capture_frame(camera):
    """
    Captura un fotograma de la cámara.

    :param camera: Objeto de la cámara.
    :return: Fotograma capturado.
    """
    ret, frame = camera.read()
    if not ret:
        raise ValueError("No se pudo capturar el fotograma de la cámara.")
    return frame

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Mostrar el fotograma
        cv2.imshow('Camera Feed', frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def capture_frame(camera):
    """
    Captura un fotograma de la cámara.

    :param camera: Objeto de la cámara.
    :return: Fotograma capturado.
    """
    ret, frame = camera.read()
    if not ret:
        raise ValueError("No se pudo capturar el fotograma de la cámara.")
    return frame

def convert_to_grayscale(image):
    """
    Convierte una imagen a escala de grises.

    :param image: Imagen en color.
    :return: Imagen en escala de grises.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Convertir el fotograma a escala de grises
        gray_frame = convert_to_grayscale(frame)
        
        # Mostrar el fotograma original
        cv2.imshow('Original Frame', frame)
        
        # Mostrar el fotograma en escala de grises
        cv2.imshow('Grayscale Frame', gray_frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



  import cv2
import numpy as np

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def capture_frame(camera):
    """
    Captura un fotograma de la cámara.

    :param camera: Objeto de la cámara.
    :return: Fotograma capturado.
    """
    ret, frame = camera.read()
    if not ret:
        raise ValueError("No se pudo capturar el fotograma de la cámara.")
    return frame

def process_image(image):
    """
    Procesa la imagen para detección de bordes y líneas.

    :param image: Imagen en color.
    :return: Imagen procesada con líneas detectadas.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detección de bordes con Canny
    edges = cv2.Canny(gray, 50, 150)

    # Detección de líneas con HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Copia de la imagen original para dibujar las líneas
    line_image = np.copy(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dibujar líneas en verde

    return line_image

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Procesar la imagen
        processed_frame = process_image(frame)
        
        # Mostrar el fotograma original
        cv2.imshow('Original Frame', frame)
        
        # Mostrar el fotograma procesado con líneas
        cv2.imshow('Processed Frame', processed_frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




  import cv2
import numpy as np

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def capture_frame(camera):
    """
    Captura un fotograma de la cámara.

    :param camera: Objeto de la cámara.
    :return: Fotograma capturado.
    """
    ret, frame = camera.read()
    if not ret:
        raise ValueError("No se pudo capturar el fotograma de la cámara.")
    return frame

def segment_image(image):
    """
    Segmenta la imagen usando umbralización y detección de contornos.

    :param image: Imagen en color.
    :return: Imagen con contornos detectados.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Umbralización de la imagen
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Detección de contornos
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copia de la imagen original para dibujar los contornos
    contour_image = np.copy(image)
    
    # Dibujar los contornos en la imagen
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Dibujar contornos en verde

    return contour_image, thresholded

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Procesar la imagen
        contour_image, thresholded = segment_image(frame)
        
        # Mostrar el fotograma original
        cv2.imshow('Original Frame', frame)
        
        # Mostrar la imagen umbralizada
        cv2.imshow('Thresholded Image', thresholded)
        
        # Mostrar la imagen con contornos detectados
        cv2.imshow('Contour Image', contour_image)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



  import cv2
import numpy as np

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    # Cargar la red YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Cargar las etiquetas de las clases
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes

def prepare_image(image):
    """
    Prepara la imagen para la detección de objetos.

    :param image: Imagen en color.
    :return: Imagen preparada para la red YOLO.
    """
    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape
    
    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    return blob, height, width

def detect_objects(net, output_layers, blob, height, width):
    """
    Realiza la detección de objetos en la imagen usando YOLO.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones.
    """
    # Realizar la detección
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar los resultados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar las cajas y etiquetas en la imagen
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    # Cargar el modelo YOLO
    net, output_layers, classes = load_yolo_model()
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Preparar la imagen para el modelo YOLO
        blob, height, width = prepare_image(frame)
        
        # Detectar objetos
        detected_frame = detect_objects(net, output_layers, blob, height, width)
        
        # Mostrar el fotograma con las detecciones
        cv2.imshow('Detected Objects', detected_frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2
import numpy as np

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    # Cargar la red YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Cargar las etiquetas de las clases
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes

def prepare_image(image):
    """
    Prepara la imagen para la detección de objetos.

    :param image: Imagen en color.
    :return: Imagen preparada para la red YOLO.
    """
    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape
    
    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    return blob, height, width

def post_process(net, output_layers, blob, height, width):
    """
    Realiza el post-procesamiento de las detecciones de objetos.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones post-procesadas.
    """
    # Realizar la detección
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar los resultados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para reducir múltiples detecciones
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Crear una imagen copia para dibujar las cajas
    output_image = np.copy(image)

    # Dibujar las cajas de las detecciones
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output_image

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    # Cargar el modelo YOLO
    net, output_layers, classes = load_yolo_model()
    
    while True:
        # Capturar un fotograma
        frame = capture_frame(camera)
        
        # Preparar la imagen para el modelo YOLO
        blob, height, width = prepare_image(frame)
        
        # Realizar el post-procesamiento
        detected_frame = post_process(net, output_layers, blob, height, width)
        
        # Mostrar el fotograma con las detecciones
        cv2.imshow('Detected Objects', detected_frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2
import numpy as np

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    # Cargar la red YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Cargar las etiquetas de las clases
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes

def prepare_image(image):
    """
    Prepara la imagen para la detección de objetos.

    :param image: Imagen en color.
    :return: Imagen preparada para la red YOLO.
    """
    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape
    
    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    return blob, height, width

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Aplica un filtro gaussiano a la imagen para reducir el ruido.

    :param image: Imagen en color.
    :param kernel_size: Tamaño del núcleo del filtro gaussiano.
    :return: Imagen filtrada.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges_sobel(image):
    """
    Detecta bordes en la imagen utilizando el operador Sobel.

    :param image: Imagen en escala de grises.
    :return: Imagen con bordes detectados.
    """
    # Aplicar Sobel en la dirección X y Y
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    
    # Convertir a valores absolutos y a escala de 8 bits
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combinar las imágenes de gradientes
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return edges

def post_process(net, output_layers, blob, height, width):
    """
    Realiza el post-procesamiento de las detecciones de objetos.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones post-procesadas.
    """
    # Realizar la detección
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar los resultados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para reducir múltiples detecciones
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Crear una imagen copia para dibujar las cajas
    output_image = np.copy(image)

    # Dibujar las cajas de las detecciones
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output_image

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    # Cargar el modelo YOLO
    net, output_layers, classes = load_yolo_model()
    
    while True:
        # Capturar un fotograma
        ret, frame = camera.read()
        if not ret:
            break

        # Aplicar filtro gaussiano para reducir ruido
        blurred_frame = apply_gaussian_blur(frame)
        
        # Convertir a escala de grises
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes usando Sobel
        edges = detect_edges_sobel(gray_frame)
        
        # Preparar la imagen para el modelo YOLO
        blob, height, width = prepare_image(frame)
        
        # Realizar el post-procesamiento
        detected_frame = post_process(net, output_layers, blob, height, width)
        
        # Mostrar el fotograma con las detecciones
        cv2.imshow('Detected Objects', detected_frame)
        
        # Mostrar el fotograma con bordes
        cv2.imshow('Edges', edges)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2
import numpy as np
import datetime

def configure_camera(camera_index=0, resolution=(1920, 1080), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    # Cargar la red YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Cargar las etiquetas de las clases
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes

def prepare_image(image):
    """
    Prepara la imagen para la detección de objetos.

    :param image: Imagen en color.
    :return: Imagen preparada para la red YOLO.
    """
    # Obtener las dimensiones de la imagen
    height, width, _ = image.shape
    
    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    return blob, height, width

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Aplica un filtro gaussiano a la imagen para reducir el ruido.

    :param image: Imagen en color.
    :param kernel_size: Tamaño del núcleo del filtro gaussiano.
    :return: Imagen filtrada.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges_sobel(image):
    """
    Detecta bordes en la imagen utilizando el operador Sobel.

    :param image: Imagen en escala de grises.
    :return: Imagen con bordes detectados.
    """
    # Aplicar Sobel en la dirección X y Y
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    
    # Convertir a valores absolutos y a escala de 8 bits
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combinar las imágenes de gradientes
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return edges

def post_process(net, output_layers, blob, height, width):
    """
    Realiza el post-procesamiento de las detecciones de objetos.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones post-procesadas.
    """
    # Realizar la detección
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar los resultados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para reducir múltiples detecciones
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Crear una imagen copia para dibujar las cajas
    output_image = np.copy(image)

    # Dibujar las cajas de las detecciones
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output_image

def save_results(image, filename):
    """
    Guarda la imagen en el disco.

    :param image: Imagen a guardar.
    :param filename: Nombre del archivo para guardar la imagen.
    """
    cv2.imwrite(filename, image)

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(1920, 1080), fps=30)
    
    # Cargar el modelo YOLO
    net, output_layers, classes = load_yolo_model()
    
    while True:
        # Capturar un fotograma
        ret, frame = camera.read()
        if not ret:
            break

        # Aplicar filtro gaussiano para reducir ruido
        blurred_frame = apply_gaussian_blur(frame)
        
        # Convertir a escala de grises
        gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes usando Sobel
        edges = detect_edges_sobel(gray_frame)
        
        # Preparar la imagen para el modelo YOLO
        blob, height, width = prepare_image(frame)
        
        # Realizar el post-procesamiento
        detected_frame = post_process(net, output_layers, blob, height, width)
        
        # Mostrar el fotograma con las detecciones
        cv2.imshow('Detected Objects', detected_frame)
        
        # Mostrar el fotograma con bordes
        cv2.imshow('Edges', edges)
        
        # Guardar resultados en disco
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(detected_frame, f'detected_objects_{timestamp}.jpg')
        save_results(edges, f'edges_{timestamp}.jpg')
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2
import numpy as np
import datetime

def configure_camera(camera_index=0, resolution=(640, 480), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
    camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
    
    return camera

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    # Cargar la red YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Cargar las etiquetas de las clases
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Obtener los nombres de las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes

def prepare_image(image, new_size=(416, 416)):
    """
    Prepara la imagen para la detección de objetos y redimensiona la imagen para aumentar la velocidad.

    :param image: Imagen en color.
    :param new_size: Tamaño nuevo para la imagen.
    :return: Imagen preparada para la red YOLO y dimensiones originales.
    """
    # Redimensionar la imagen para aumentar la velocidad de procesamiento
    resized_image = cv2.resize(image, new_size)
    
    # Obtener las dimensiones de la imagen
    height, width, _ = resized_image.shape
    
    # Preparar la imagen para el modelo YOLO
    blob = cv2.dnn.blobFromImage(resized_image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    return blob, height, width

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Aplica un filtro gaussiano a la imagen para reducir el ruido.

    :param image: Imagen en color.
    :param kernel_size: Tamaño del núcleo del filtro gaussiano.
    :return: Imagen filtrada.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges_sobel(image):
    """
    Detecta bordes en la imagen utilizando el operador Sobel.

    :param image: Imagen en escala de grises.
    :return: Imagen con bordes detectados.
    """
    # Aplicar Sobel en la dirección X y Y
    grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    
    # Convertir a valores absolutos y a escala de 8 bits
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # Combinar las imágenes de gradientes
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return edges

def post_process(net, output_layers, blob, height, width):
    """
    Realiza el post-procesamiento de las detecciones de objetos.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones post-procesadas.
    """
    # Realizar la detección
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar los resultados
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Umbral de confianza
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression (NMS) para reducir múltiples detecciones
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Crear una imagen copia para dibujar las cajas
    output_image = np.copy(image)

    # Dibujar las cajas de las detecciones
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output_image

def save_results(image, filename):
    """
    Guarda la imagen en el disco.

    :param image: Imagen a guardar.
    :param filename: Nombre del archivo para guardar la imagen.
    """
    cv2.imwrite(filename, image)

def process_images_batch(images):
    """
    Procesa un lote de imágenes utilizando la red YOLO.

    :param images: Lista de imágenes en color.
    :return: Lista de imágenes con detecciones post-procesadas.
    """
    # Cargar el modelo YOLO
    net, output_layers, classes = load_yolo_model()
    
    results = []
    for image in images:
        # Preparar la imagen para el modelo YOLO
        blob, height, width = prepare_image(image)
        
        # Realizar el post-procesamiento
        detected_image = post_process(net, output_layers, blob, height, width)
        
        results.append(detected_image)
    
    return results

def main():
    # Configurar la cámara
    camera = configure_camera(camera_index=0, resolution=(640, 480), fps=30)
    
    batch_size = 5
    image_batch = []
    
    while True:
        # Capturar un fotograma
        ret, frame = camera.read()
        if not ret:
            break

        # Agregar el fotograma al lote
        image_batch.append(frame)
        
        # Procesar el lote si alcanzamos el tamaño del lote
        if len(image_batch) >= batch_size:
            # Procesar imágenes en batch
            detected_images = process_images_batch(image_batch)
            
            # Guardar y mostrar resultados para cada imagen procesada
            for idx, detected_image in enumerate(detected_images):
                # Guardar resultados en disco
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
            
            # Limpiar el lote
            image_batch = []
        
        # Convertir a escala de grises para bordes
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes usando Sobel
        edges = detect_edges_sobel(gray_frame)
        
        # Mostrar el fotograma con bordes
        cv2.imshow('Edges', edges)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Procesar cualquier imagen restante en el lote
    if image_batch:
        detected_images = process_images_batch(image_batch)
        for idx, detected_image in enumerate(detected_images):
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
    
    # Liberar la cámara y cerrar las ventanas
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


  import cv2
import numpy as np
import datetime

# Simulación de sensores
def get_sensor_data():
    """
    Simula la adquisición de datos de sensores, en este caso, genera una imagen de prueba.

    :return: Imagen de prueba como un array numpy.
    """
    # Crear una imagen de prueba (en un caso real, esto vendría de los sensores)
    image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    return image

def configure_camera(camera_index=0, resolution=(640, 480), fps=30):
    """
    Configura la cámara con el índice, resolución y FPS deseados.

    :param camera_index: Índice de la cámara (0 por defecto para la cámara principal).
    :param resolution: Resolución deseada (anchura, altura).
    :param fps: Tasa de fotogramas por segundo.
    :return: Objeto de la cámara configurado.
    """
    try:
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # Ajustar la anchura del marco
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # Ajustar la altura del marco
        camera.set(cv2.CAP_PROP_FPS, fps)  # Ajustar la tasa de fotogramas
        
        return camera
    except Exception as e:
        print(f"Error en la configuración de la cámara: {e}")
        raise

def load_yolo_model():
    """
    Carga el modelo YOLO preentrenado.

    :return: Red YOLO, nombres de las clases y colores para dibujar.
    """
    try:
        # Cargar la red YOLO
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        
        # Cargar las etiquetas de las clases
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # Obtener los nombres de las capas de salida
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return net, output_layers, classes
    except Exception as e:
        print(f"Error al cargar el modelo YOLO: {e}")
        raise

def prepare_image(image, new_size=(416, 416)):
    """
    Prepara la imagen para la detección de objetos y redimensiona la imagen para aumentar la velocidad.

    :param image: Imagen en color.
    :param new_size: Tamaño nuevo para la imagen.
    :return: Imagen preparada para la red YOLO y dimensiones originales.
    """
    try:
        # Redimensionar la imagen para aumentar la velocidad de procesamiento
        resized_image = cv2.resize(image, new_size)
        
        # Obtener las dimensiones de la imagen
        height, width, _ = resized_image.shape
        
        # Preparar la imagen para el modelo YOLO
        blob = cv2.dnn.blobFromImage(resized_image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
        return blob, height, width
    except Exception as e:
        print(f"Error al preparar la imagen: {e}")
        raise

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Aplica un filtro gaussiano a la imagen para reducir el ruido.

    :param image: Imagen en color.
    :param kernel_size: Tamaño del núcleo del filtro gaussiano.
    :return: Imagen filtrada.
    """
    try:
        return cv2.GaussianBlur(image, kernel_size, 0)
    except Exception as e:
        print(f"Error al aplicar el filtro gaussiano: {e}")
        raise

def detect_edges_sobel(image):
    """
    Detecta bordes en la imagen utilizando el operador Sobel.

    :param image: Imagen en escala de grises.
    :return: Imagen con bordes detectados.
    """
    try:
        # Aplicar Sobel en la dirección X y Y
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        
        # Convertir a valores absolutos y a escala de 8 bits
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # Combinar las imágenes de gradientes
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        return edges
    except Exception as e:
        print(f"Error al detectar bordes con Sobel: {e}")
        raise

def post_process(net, output_layers, blob, height, width):
    """
    Realiza el post-procesamiento de las detecciones de objetos.

    :param net: Red YOLO.
    :param output_layers: Capas de salida de la red.
    :param blob: Imagen preparada.
    :param height: Altura original de la imagen.
    :param width: Anchura original de la imagen.
    :return: Imagen con detecciones post-procesadas.
    """
    try:
        # Realizar la detección
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Procesar los resultados
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:  # Umbral de confianza
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression (NMS) para reducir múltiples detecciones
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Crear una imagen copia para dibujar las cajas
        output_image = np.copy(image)

        # Dibujar las cajas de las detecciones
        colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    except Exception as e:
        print(f"Error en el post-procesamiento de la imagen: {e}")
        raise

def save_results(image, filename):
    """
    Guarda la imagen en el disco.

    :param image: Imagen a guardar.
    :param filename: Nombre del archivo para guardar la imagen.
    """
    try:
        cv2.imwrite(filename, image)
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        raise

def process_images_batch(images):
    """
    Procesa un lote de imágenes utilizando la red YOLO.

    :param images: Lista de imágenes en color.
    :return: Lista de imágenes con detecciones post-procesadas.
    """
    try:
        # Cargar el modelo YOLO
        net, output_layers, classes = load_yolo_model()
        
        results = []
        for image in images:
            # Preparar la imagen para el modelo YOLO
            blob, height, width = prepare_image(image)
            
            # Realizar el post-procesamiento
            detected_image = post_process(net, output_layers, blob, height, width)
            
            results.append(detected_image)
        
        return results
    except Exception as e:
        print(f"Error al procesar el lote de imágenes: {e}")
        raise

def test_apply_gaussian_blur():
    """
    Prueba unitaria para la función de aplicación de filtro gaussiano.
    """
    try:
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        blurred_image = apply_gaussian_blur(test_image)
        assert blurred_image.shape == test_image.shape, "Error: El tamaño de la imagen no coincide"
        print("Prueba de filtro gaussiano pasada")
    except AssertionError as e:
        print(f"Error en la prueba de filtro gaussiano: {e}")
    except Exception as e:
        print(f"Error en la prueba de filtro gaussiano: {e}")

def test_detect_edges_sobel():
    """
    Prueba unitaria para la función de detección de bordes con Sobel.
    """
    try:
        test_image = np.random.randint(0, 255, (640, 480), dtype=np.uint8)
        edges_image = detect_edges_sobel(test_image)
        assert edges_image.shape == test_image.shape, "Error: El tamaño de la imagen no coincide"
        print("Prueba de detección de bordes pasada")
    except AssertionError as e:
        print(f"Error en la prueba de detección de bordes: {e}")
    except Exception as e:
        print(f"Error en la prueba de detección de bordes: {e}")

def main():
    try:
        # Configurar la cámara
        camera = configure_camera(camera_index=0, resolution=(640, 480), fps=30)
        
        batch_size = 5
        image_batch = []

        while True:
            # Capturar un fotograma
            ret, frame = camera.read()
            if not ret:
                break

            # Agregar el fotograma al lote
            image_batch.append(frame)
            
            # Procesar el lote si alcanzamos el tamaño del lote
            if len(image_batch) >= batch_size:
                # Procesar imágenes en batch
                detected_images = process_images_batch(image_batch)
                
                # Guardar y mostrar resultados para cada imagen procesada
                for idx, detected_image in enumerate(detected_images):
                    # Guardar resultados en disco
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
                
                # Limpiar el lote
                image_batch = []
            
            # Convertir a escala de grises para bordes
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordes usando Sobel
            edges = detect_edges_sobel(gray_frame)
            
            # Mostrar el fotograma con bordes
            cv2.imshow('Edges', edges)
            
            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Procesar cualquier imagen restante en el lote
        if image_batch:
            detected_images = process_images_batch(image_batch)
            for idx, detected_image in enumerate(detected_images):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
        
        # Liberar la cámara y cerrar las ventanas
        camera.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error en el proceso principal: {e}")

if __name__ == "__main__":
    # Ejecutar pruebas unitarias
    test_apply_gaussian_blur()
    test_detect_edges_sobel()
    
    # Ejecutar el flujo principal
    main()


  import cv2
import numpy as np
import datetime
import socket
import threading
import tkinter as tk
from tkinter import Label, PhotoImage
import logging

# Configuración de logging
logging.basicConfig(filename='system.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funciones de procesamiento (igual que antes)

def configure_camera(camera_index=0, resolution=(640, 480), fps=30):
    try:
        camera = cv2.VideoCapture(camera_index)
        
        if not camera.isOpened():
            raise ValueError(f"No se pudo abrir la cámara con índice {camera_index}.")
        
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        camera.set(cv2.CAP_PROP_FPS, fps)
        
        return camera
    except Exception as e:
        logging.error(f"Error en la configuración de la cámara: {e}")
        raise

def load_yolo_model():
    try:
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return net, output_layers, classes
    except Exception as e:
        logging.error(f"Error al cargar el modelo YOLO: {e}")
        raise

def prepare_image(image, new_size=(416, 416)):
    try:
        resized_image = cv2.resize(image, new_size)
        height, width, _ = resized_image.shape
        blob = cv2.dnn.blobFromImage(resized_image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
        return blob, height, width
    except Exception as e:
        logging.error(f"Error al preparar la imagen: {e}")
        raise

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    try:
        return cv2.GaussianBlur(image, kernel_size, 0)
    except Exception as e:
        logging.error(f"Error al aplicar el filtro gaussiano: {e}")
        raise

def detect_edges_sobel(image):
    try:
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        return edges
    except Exception as e:
        logging.error(f"Error al detectar bordes con Sobel: {e}")
        raise

def post_process(net, output_layers, blob, height, width):
    try:
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(obj[0] * width)
                        center_y = int(obj[1] * height)
                        w = int(obj[2] * width)
                        h = int(obj[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        output_image = np.copy(image)
        colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_image
    except Exception as e:
        logging.error(f"Error en el post-procesamiento de la imagen: {e}")
        raise

def save_results(image, filename):
    try:
        cv2.imwrite(filename, image)
    except Exception as e:
        logging.error(f"Error al guardar la imagen: {e}")
        raise

def process_images_batch(images):
    try:
        net, output_layers, classes = load_yolo_model()
        
        results = []
        for image in images:
            blob, height, width = prepare_image(image)
            detected_image = post_process(net, output_layers, blob, height, width)
            results.append(detected_image)
        
        return results
    except Exception as e:
        logging.error(f"Error al procesar el lote de imágenes: {e}")
        raise

def test_apply_gaussian_blur():
    try:
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        blurred_image = apply_gaussian_blur(test_image)
        assert blurred_image.shape == test_image.shape, "Error: El tamaño de la imagen no coincide"
        logging.info("Prueba de filtro gaussiano pasada")
    except AssertionError as e:
        logging.error(f"Error en la prueba de filtro gaussiano: {e}")
    except Exception as e:
        logging.error(f"Error en la prueba de filtro gaussiano: {e}")

def test_detect_edges_sobel():
    try:
        test_image = np.random.randint(0, 255, (640, 480), dtype=np.uint8)
        edges_image = detect_edges_sobel(test_image)
        assert edges_image.shape == test_image.shape, "Error: El tamaño de la imagen no coincide"
        logging.info("Prueba de detección de bordes pasada")
    except AssertionError as e:
        logging.error(f"Error en la prueba de detección de bordes: {e}")
    except Exception as e:
        logging.error(f"Error en la prueba de detección de bordes: {e}")

# Comunicación de Resultados
def send_results_by_network(image, server_address=('localhost', 5000)):
    """
    Envía la imagen procesada a través de una conexión de red.

    :param image: Imagen procesada.
    :param server_address: Dirección del servidor para enviar la imagen.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(server_address)
        
        # Convertir la imagen a formato JPEG
        _, buffer = cv2.imencode('.jpg', image)
        data = buffer.tobytes()

        # Enviar tamaño de datos y datos de la imagen
        s.sendall(len(data).to_bytes(4, 'big'))
        s.sendall(data)
        
        s.close()
        logging.info("Imagen enviada por red")
    except Exception as e:
        logging.error(f"Error al enviar la imagen por red: {e}")

# Interfaz de Usuario con Tkinter
class RealTimeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualización en Tiempo Real")
        
        self.image_label = Label(root)
        self.image_label.pack()

        self.update_interval = 20  # ms
        self.update()

    def update(self):
        try:
            # Capturar imagen de prueba (en un caso real, capturar de la cámara)
            image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            # Convertir imagen a formato compatible con Tkinter
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = PhotoImage(image=Image.fromarray(image_rgb))
            
            self.image_label.config(image=image_pil)
            self.image_label.image = image_pil

            # Actualizar cada `update_interval` milisegundos
            self.root.after(self.update_interval, self.update)
        except Exception as e:
            logging.error(f"Error en la actualización de la interfaz de usuario: {e}")

def main():
    try:
        # Configurar la cámara
        camera = configure_camera(camera_index=0, resolution=(640, 480), fps=30)
        
        batch_size = 5
        image_batch = []

        # Configurar interfaz de usuario
        root = tk.Tk()
        ui = RealTimeUI(root)
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Agregar el fotograma al lote
            image_batch.append(frame)
            
            # Procesar el lote si alcanzamos el tamaño del lote
            if len(image_batch) >= batch_size:
                # Procesar imágenes en batch
                detected_images = process_images_batch(image_batch)
                
                # Guardar y mostrar resultados para cada imagen procesada
                for idx, detected_image in enumerate(detected_images):
                    # Guardar resultados en disco
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
                    
                    # Enviar resultados por red
                    send_results_by_network(detected_image)
                
                # Limpiar el lote
                image_batch = []
            
            # Convertir a escala de grises para bordes
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar bordes usando Sobel
            edges = detect_edges_sobel(gray_frame)
            
            # Mostrar el fotograma con bordes
            cv2.imshow('Edges', edges)
            
            # Actualizar interfaz de usuario
            ui.update()
            
            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Procesar cualquier imagen restante en el lote
        if image_batch:
            detected_images = process_images_batch(image_batch)
            for idx, detected_image in enumerate(detected_images):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                save_results(detected_image, f'detected_objects_{timestamp}_{idx}.jpg')
                send_results_by_network(detected_image)
        
        # Liberar la cámara y cerrar las ventanas
        camera.release()
        cv2.destroyAllWindows()
        root.mainloop()
    
    except Exception as e:
        logging.error(f"Error en el proceso principal: {e}")

if __name__ == "__main__":
    # Ejecutar pruebas unitarias
    test_apply_gaussian_blur()
    test_detect_edges_sobel()
    
    # Ejecutar el flujo principal
    main()
