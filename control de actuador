import RPi.GPIO as GPIO
import time
from Adafruit_PCA9685 import PCA9685
import serial

# Configuración de pines y parámetros
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Mínimo pulso (ajusta según tu servomotor)
servo_max = 600  # Máximo pulso (ajusta según tu servomotor)
sensor_pin = 17  # Pin GPIO para el sensor (ajusta según tu configuración)
threshold = 500  # Umbral para detección de cable muerto

# Configuración del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Frecuencia en Hz

# Configuración del puerto serial
serial_port = '/dev/ttyUSB0'  # Ajusta según el puerto serial que estés usando
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)

# Configuración de GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)

def set_servo_angle(channel, angle):
    """Configura el ángulo del servomotor."""
    pulse = servo_min + (servo_max - servo_min) * (angle / 180.0)
    pwm.set_pwm(channel, 0, int(pulse))

def cut_cable():
    """Activa el servomotor para cortar el cable y luego regresa a la posición de reposo."""
    print("Cable muerto detectado. Cortando cable...")
    set_servo_angle(servo_channel, 90)  # Ajusta el ángulo según tu sistema
    time.sleep(2)  # Tiempo para cortar el cable
    set_servo_angle(servo_channel, 0)  # Regresa el servomotor a la posición de reposo

def main():
    """Función principal del programa."""
    try:
        while True:
            # Lee el valor del sensor
            sensor_value = GPIO.input(sensor_pin)
            print(f"Sensor Value: {sensor_value}")
            
            # Compara el valor del sensor con el umbral
            if sensor_value < threshold:
                cut_cable()
            else:
                print("Cable activo. Actuador en reposo.")
                set_servo_angle(servo_channel, 0)  # Asegúrate de que el servomotor esté en reposo
            
            time.sleep(1)  # Espera antes de la siguiente lectura
    except KeyboardInterrupt:
        print("Interrupción del usuario. Limpiando GPIO...")
    finally:
        GPIO.cleanup()  # Limpia la configuración de GPIO

if __name__ == "__main__":
    main()


import RPi.GPIO as GPIO
import time
from Adafruit_PCA9685 import PCA9685

# Configuración de pines y parámetros
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)
sensor_pin = 17  # Pin GPIO para el sensor (ajusta según tu configuración)
threshold = 500  # Umbral para detección de cable muerto

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

# Configuración de GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)  # Configura el pin del sensor como entrada

def set_servo_angle(channel, angle):
    """Configura el ángulo del servomotor."""
    pulse = servo_min + (servo_max - servo_min) * (angle / 180.0)
    pwm.set_pwm(channel, 0, int(pulse))

def main():
    """Función principal del programa."""
    try:
        # Inicializa el servomotor en la posición de reposo
        print("Inicializando el servomotor...")
        set_servo_angle(servo_channel, 0)  # Posición inicial del servomotor

        while True:
            # Lee el valor del sensor
            sensor_value = GPIO.input(sensor_pin)
            print(f"Sensor Value: {sensor_value}")

            # Compara el valor del sensor con el umbral
            if sensor_value < threshold:
                print("Cable muerto detectado. Cortando cable...")
                set_servo_angle(servo_channel, 90)  # Ajusta el ángulo según tu sistema
                time.sleep(2)  # Tiempo para cortar el cable
                set_servo_angle(servo_channel, 0)  # Regresa el servomotor a la posición de reposo
            else:
                print("Cable activo. Actuador en reposo.")
                set_servo_angle(servo_channel, 0)  # Asegúrate de que el servomotor esté en reposo
            
            time.sleep(1)  # Espera antes de la siguiente lectura

    except KeyboardInterrupt:
        print("Interrupción del usuario. Limpiando GPIO...")
    finally:
        GPIO.cleanup()  # Limpia la configuración de GPIO

if __name__ == "__main__":
    main()


import time
from Adafruit_PCA9685 import PCA9685

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_angle(channel, angle):
    """Configura el ángulo del servomotor."""
    pulse = servo_min + (servo_max - servo_min) * (angle / 180.0)
    pwm.set_pwm(channel, 0, int(pulse))

def move_actuator_to_position(angle):
    """Mueve el actuador a una posición específica."""
    print(f"Moviendo actuador a {angle} grados...")
    set_servo_angle(servo_channel, angle)
    time.sleep(1)  # Espera a que el servomotor llegue a la posición

def move_actuator_increments(start_angle, end_angle, step, delay):
    """Mueve el actuador en incrementos desde una posición inicial hasta una final."""
    if start_angle < end_angle:
        for angle in range(start_angle, end_angle + 1, step):
            print(f"Moviendo actuador a {angle} grados...")
            set_servo_angle(servo_channel, angle)
            time.sleep(delay)
    elif start_angle > end_angle:
        for angle in range(start_angle, end_angle - 1, -step):
            print(f"Moviendo actuador a {angle} grados...")
            set_servo_angle(servo_channel, angle)
            time.sleep(delay)
    else:
        print("La posición inicial y final son iguales.")

def main():
    """Función principal para probar los movimientos del actuador."""
    try:
        # Mueve el actuador a una posición específica
        move_actuator_to_position(90)  # Ajusta el ángulo según tu necesidad
        time.sleep(2)

        # Mueve el actuador en incrementos
        move_actuator_increments(0, 180, 10, 0.5)  # Ajusta los parámetros según tu necesidad
        time.sleep(2)
        move_actuator_increments(180, 0, 10, 0.5)  # Regresa a la posición inicial

    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        pwm.set_pwm(servo_channel, 0, 0)  # Asegúrate de que el servomotor esté en reposo

if __name__ == "__main__":
    main()



import serial
import time

# Configuración del puerto serial
serial_port = '/dev/ttyUSB0'  # Ajusta según el puerto serial que estés usando
baud_rate = 9600  # Velocidad de baudios
timeout = 1  # Tiempo de espera para la comunicación serial en segundos

# Inicialización de la comunicación serial
ser = serial.Serial(serial_port, baud_rate, timeout=timeout)

def send_command(command):
    """Envía un comando al actuador a través del puerto serial."""
    if ser.is_open:
        print(f"Enviando comando: {command}")
        ser.write(command.encode())  # Envía el comando codificado como bytes
        time.sleep(0.5)  # Espera para permitir que el actuador procese el comando
    else:
        print("El puerto serial no está abierto.")

def main():
    """Función principal para enviar comandos al actuador."""
    try:
        while True:
            # Ejemplo de comandos para mover el actuador
            send_command('MOVE_TO_90')  # Envía un comando para mover el actuador a 90 grados
            time.sleep(5)  # Espera antes de enviar el siguiente comando

            send_command('MOVE_TO_0')  # Envía un comando para mover el actuador a 0 grados
            time.sleep(5)  # Espera antes de enviar el siguiente comando

    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        ser.close()  # Cierra el puerto serial cuando se interrumpe el programa

if __name__ == "__main__":
    main()



import time
from Adafruit_PCA9685 import PCA9685

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def calibrate_actuator():
    """Calibra el actuador moviéndolo a las posiciones mínima y máxima."""
    print("Comenzando calibración del actuador...")
    
    # Mueve el actuador a la posición mínima
    print("Moviendo a la posición mínima...")
    set_servo_pulse(servo_channel, servo_min)
    time.sleep(2)  # Espera para permitir que el servomotor alcance la posición
    
    # Mueve el actuador a la posición máxima
    print("Moviendo a la posición máxima...")
    set_servo_pulse(servo_channel, servo_max)
    time.sleep(2)  # Espera para permitir que el servomotor alcance la posición
    
    # Mueve el actuador a la posición media
    mid_pulse = (servo_min + servo_max) // 2
    print("Moviendo a la posición media...")
    set_servo_pulse(servo_channel, mid_pulse)
    time.sleep(2)  # Espera para permitir que el servomotor alcance la posición
    
    print("Calibración completa.")

def main():
    """Función principal para calibrar el actuador."""
    try:
        calibrate_actuator()
    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        # Asegúrate de que el servomotor esté en una posición segura antes de finalizar
        set_servo_pulse(servo_channel, servo_min)
        print("Actuador en posición mínima.")

if __name__ == "__main__":
    main()



import time
import RPi.GPIO as GPIO
from Adafruit_PCA9685 import PCA9685

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Configuración del sensor de distancia
trigger_pin = 23  # Pin GPIO para el trigger del sensor ultrasonido
echo_pin = 24     # Pin GPIO para el echo del sensor ultrasonido
obstacle_threshold = 10  # Umbral de distancia en cm para detectar obstáculos

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

# Configuración de GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def get_distance():
    """Obtiene la distancia del sensor ultrasonido en cm."""
    # Envia un pulso de 10us para iniciar la medición
    GPIO.output(trigger_pin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)
    
    # Espera a que el pin de eco se vuelva alto
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
    
    # Espera a que el pin de eco se vuelva bajo
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
    
    # Calcula la duración del pulso
    pulse_duration = pulse_end - pulse_start
    
    # Calcula la distancia en cm
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return distance

def move_actuator_safe(angle):
    """Mueve el actuador a una posición segura, comprobando obstáculos."""
    print(f"Verificando obstáculos antes de mover a {angle} grados...")
    distance = get_distance()
    print(f"Distancia medida: {distance} cm")
    
    if distance < obstacle_threshold:
        print("¡Obstáculo detectado! Movimiento detenido para evitar daños.")
        set_servo_pulse(servo_channel, servo_min)  # Posición segura en caso de obstáculo
    else:
        print("No se detectaron obstáculos. Moviendo actuador...")
        set_servo_pulse(servo_channel, servo_min + (servo_max - servo_min) * (angle / 180.0))
        time.sleep(2)  # Espera para permitir que el servomotor alcance la posición

def main():
    """Función principal para probar el movimiento seguro del actuador."""
    try:
        # Mueve el actuador a una posición segura
        move_actuator_safe(90)  # Ajusta el ángulo según tu necesidad
        time.sleep(5)
        move_actuator_safe(0)  # Regresa a la posición inicial
        time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        GPIO.cleanup()  # Limpia la configuración de GPIO

if __name__ == "__main__":
    main()



import time
from Adafruit_PCA9685 import PCA9685
from simple_pid import PID

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

# Configuración del PID
pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=90)
pid.output_limits = (servo_min, servo_max)  # Limita la salida del PID al rango del servomotor

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def get_current_position():
    """Obtiene la posición actual del servomotor (dummy function)."""
    # Esta función debe ser reemplazada con una lectura real del sensor de posición si está disponible.
    # Aquí asumimos que la posición actual es fija para simplificar el ejemplo.
    return 90  # Valor dummy; reemplaza con lectura real si está disponible

def control_actuator_pid(desired_position):
    """Controla el actuador utilizando el controlador PID."""
    pid.setpoint = desired_position  # Establece la posición deseada
    while True:
        current_position = get_current_position()  # Obtiene la posición actual
        control_signal = pid(current_position)  # Calcula la señal de control usando PID
        print(f"Posición actual: {current_position}, Señal de control: {control_signal}")
        set_servo_pulse(servo_channel, control_signal)  # Ajusta la posición del servomotor
        time.sleep(0.1)  # Espera un breve período antes de la siguiente iteración

def main():
    """Función principal para controlar el actuador con PID."""
    try:
        # Mueve el actuador a la posición deseada utilizando PID
        control_actuator_pid(90)  # Ajusta el ángulo deseado según tu necesidad
    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        set_servo_pulse(servo_channel, servo_min)  # Asegúrate de que el servomotor esté en reposo

if __name__ == "__main__":
    main()


import time
import RPi.GPIO as GPIO
from Adafruit_PCA9685 import PCA9685

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Configuración del sensor de distancia
trigger_pin = 23  # Pin GPIO para el trigger del sensor ultrasonido
echo_pin = 24     # Pin GPIO para el echo del sensor ultrasonido
distance_threshold = 20  # Umbral de distancia en cm para ajustar el actuador

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

# Configuración de GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def get_distance():
    """Obtiene la distancia del sensor ultrasonido en cm."""
    GPIO.output(trigger_pin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(trigger_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, GPIO.LOW)
    
    while GPIO.input(echo_pin) == GPIO.LOW:
        pulse_start = time.time()
    
    while GPIO.input(echo_pin) == GPIO.HIGH:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return distance

def control_actuator_based_on_sensor():
    """Controla el actuador basado en la lectura del sensor."""
    while True:
        distance = get_distance()
        print(f"Distancia medida: {distance} cm")
        
        if distance < distance_threshold:
            # Ajusta el servomotor en función de la distancia medida
            position = servo_min + (servo_max - servo_min) * (distance / distance_threshold)
            position = min(max(position, servo_min), servo_max)  # Asegúrate de que la posición esté dentro del rango
            print(f"Ajustando actuador a la posición: {position}")
            set_servo_pulse(servo_channel, int(position))
        else:
            print("Distancia fuera del umbral, actuador no ajustado.")
        
        time.sleep(1)  # Espera antes de la siguiente lectura del sensor

def main():
    """Función principal para controlar el actuador basado en los sensores."""
    try:
        control_actuator_based_on_sensor()
    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        GPIO.cleanup()  # Limpia la configuración de GPIO

if __name__ == "__main__":
    main()



import time
import RPi.GPIO as GPIO
from Adafruit_PCA9685 import PCA9685

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Configuración de movimientos automáticos
positions = [0, 45, 90, 135, 180]  # Lista de posiciones a las que el actuador se moverá
delay_between_movements = 2  # Tiempo en segundos entre movimientos

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def calculate_servo_position(angle):
    """Calcula el pulso del servomotor en función del ángulo deseado."""
    return int(servo_min + (servo_max - servo_min) * (angle / 180.0))

def automate_actuator_movements():
    """Automatiza los movimientos del actuador a través de una serie de posiciones predefinidas."""
    while True:
        for position in positions:
            pulse = calculate_servo_position(position)
            print(f"Moviendo actuador a la posición: {position} grados (Pulso: {pulse})")
            set_servo_pulse(servo_channel, pulse)
            time.sleep(delay_between_movements)  # Espera antes del próximo movimiento

def main():
    """Función principal para iniciar la automatización de los movimientos del actuador."""
    try:
        automate_actuator_movements()
    except KeyboardInterrupt:
        print("Interrupción del usuario.")
    finally:
        set_servo_pulse(servo_channel, calculate_servo_position(0))  # Regresa el servomotor a la posición inicial

if __name__ == "__main__":
    main()


from flask import Flask, request, jsonify
from Adafruit_PCA9685 import PCA9685
import RPi.GPIO as GPIO

app = Flask(__name__)

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def calculate_servo_position(angle):
    """Calcula el pulso del servomotor en función del ángulo deseado."""
    return int(servo_min + (servo_max - servo_min) * (angle / 180.0))

@app.route('/')
def index():
    return '''
        <h1>Control Remoto del Actuador</h1>
        <form action="/move" method="post">
            <label for="angle">Ángulo (0-180):</label>
            <input type="number" id="angle" name="angle" min="0" max="180" required>
            <input type="submit" value="Mover Actuador">
        </form>
    '''

@app.route('/move', methods=['POST'])
def move_actuator():
    try:
        angle = int(request.form['angle'])
        if 0 <= angle <= 180:
            pulse = calculate_servo_position(angle)
            set_servo_pulse(servo_channel, pulse)
            return jsonify({"status": "success", "message": f"Actuador movido a {angle} grados"}), 200
        else:
            return jsonify({"status": "error", "message": "Ángulo fuera de rango"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Asegúrate de cambiar el puerto si es necesario


from flask import Flask, request, jsonify
from Adafruit_PCA9685 import PCA9685
import time

app = Flask(__name__)

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)
movement_speed = 0.1  # Tiempo en segundos entre incrementos para optimizar la velocidad

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def calculate_servo_position(angle):
    """Calcula el pulso del servomotor en función del ángulo deseado."""
    return int(servo_min + (servo_max - servo_min) * (angle / 180.0))

def move_servo_smoothly(start_angle, end_angle):
    """Mueve el servomotor de forma suave entre dos ángulos."""
    start_pulse = calculate_servo_position(start_angle)
    end_pulse = calculate_servo_position(end_angle)
    steps = abs(end_angle - start_angle) // 1  # Número de pasos basados en ángulo de 1 grado
    for step in range(steps + 1):
        angle = start_angle + step * (end_angle - start_angle) / steps
        pulse = calculate_servo_position(angle)
        set_servo_pulse(servo_channel, pulse)
        time.sleep(movement_speed)

@app.route('/')
def index():
    return '''
        <h1>Control Optimizado del Actuador</h1>
        <form action="/move" method="post">
            <label for="start_angle">Ángulo Inicial (0-180):</label>
            <input type="number" id="start_angle" name="start_angle" min="0" max="180" required>
            <br>
            <label for="end_angle">Ángulo Final (0-180):</label>
            <input type="number" id="end_angle" name="end_angle" min="0" max="180" required>
            <br>
            <input type="submit" value="Mover Actuador">
        </form>
    '''

@app.route('/move', methods=['POST'])
def move_actuator():
    try:
        start_angle = int(request.form['start_angle'])
        end_angle = int(request.form['end_angle'])
        if 0 <= start_angle <= 180 and 0 <= end_angle <= 180:
            move_servo_smoothly(start_angle, end_angle)
            return jsonify({"status": "success", "message": f"Actuador movido de {start_angle} a {end_angle} grados"}), 200
        else:
            return jsonify({"status": "error", "message": "Ángulo fuera de rango"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Asegúrate de cambiar el puerto si es necesario



from flask import Flask, request, jsonify
from Adafruit_PCA9685 import PCA9685
import time
import logging

app = Flask(__name__)

# Configuración del servomotor
servo_channel = 0  # Canal del servomotor en el PCA9685
servo_min = 150  # Pulso mínimo para el servomotor (ajusta según tu servomotor)
servo_max = 600  # Pulso máximo para el servomotor (ajusta según tu servomotor)
movement_speed = 0.1  # Tiempo en segundos entre incrementos para optimizar la velocidad

# Configuración del logging
logging.basicConfig(filename='actuator_movements.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialización del PCA9685
pwm = PCA9685()
pwm.set_pwm_freq(50)  # Configura la frecuencia del PWM a 50 Hz

def set_servo_pulse(channel, pulse):
    """Configura el pulso del servomotor en función del canal y el pulso dado."""
    pwm.set_pwm(channel, 0, pulse)

def calculate_servo_position(angle):
    """Calcula el pulso del servomotor en función del ángulo deseado."""
    return int(servo_min + (servo_max - servo_min) * (angle / 180.0))

def move_servo_smoothly(start_angle, end_angle):
    """Mueve el servomotor de forma suave entre dos ángulos y registra el movimiento."""
    start_pulse = calculate_servo_position(start_angle)
    end_pulse = calculate_servo_position(end_angle)
    steps = abs(end_angle - start_angle) // 1  # Número de pasos basados en ángulo de 1 grado
    for step in range(steps + 1):
        angle = start_angle + step * (end_angle - start_angle) / steps
        pulse = calculate_servo_position(angle)
        set_servo_pulse(servo_channel, pulse)
        logging.info(f"Moved actuator to angle {angle} degrees (Pulse: {pulse})")
        time.sleep(movement_speed)

@app.route('/')
def index():
    return '''
        <h1>Control y Monitoreo del Actuador</h1>
        <form action="/move" method="post">
            <label for="start_angle">Ángulo Inicial (0-180):</label>
            <input type="number" id="start_angle" name="start_angle" min="0" max="180" required>
            <br>
            <label for="end_angle">Ángulo Final (0-180):</label>
            <input type="number" id="end_angle" name="end_angle" min="0" max="180" required>
            <br>
            <input type="submit" value="Mover Actuador">
        </form>
        <br>
        <a href="/status">Ver Estado del Actuador</a>
    '''

@app.route('/move', methods=['POST'])
def move_actuator():
    try:
        start_angle = int(request.form['start_angle'])
        end_angle = int(request.form['end_angle'])
        if 0 <= start_angle <= 180 and 0 <= end_angle <= 180:
            move_servo_smoothly(start_angle, end_angle)
            return jsonify({"status": "success", "message": f"Actuador movido de {start_angle} a {end_angle} grados"}), 200
        else:
            return jsonify({"status": "error", "message": "Ángulo fuera de rango"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/status')
def status():
    """Muestra el estado actual del actuador."""
    try:
        # Obtén la posición actual del servomotor, para este ejemplo simplemente retornamos una estimación
        # En un caso real, puedes agregar lógica para obtener el estado real del actuador
        current_position = "Desconocida"  # Modifica según tu lógica
        return f'<h1>Estado Actual del Actuador</h1><p>Posición Actual: {current_position}</p>'
    except Exception as e:
        return f'<h1>Error</h1><p>{str(e)}</p>'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Asegúrate de cambiar el puerto si es necesario


# actuator_control.py
from Adafruit_PCA9685 import PCA9685

class ActuatorControl:
    def __init__(self):
        self.servo_channel = 0
        self.servo_min = 150
        self.servo_max = 600
        self.pwm = PCA9685()
        self.pwm.set_pwm_freq(50)
    
    def set_servo_pulse(self, channel, pulse):
        """Configura el pulso del servomotor en función del canal y el pulso dado."""
        self.pwm.set_pwm(channel, 0, pulse)
    
    def calculate_servo_position(self, angle):
        """Calcula el pulso del servomotor en función del ángulo deseado."""
        return int(self.servo_min + (self.servo_max - self.servo_min) * (angle / 180.0))
    
    def move_servo_smoothly(self, start_angle, end_angle, movement_speed):
        """Mueve el servomotor de forma suave entre dos ángulos."""
        start_pulse = self.calculate_servo_position(start_angle)
        end_pulse = self.calculate_servo_position(end_angle)
        steps = abs(end_angle - start_angle) // 1
        for step in range(steps + 1):
            angle = start_angle + step * (end_angle - start_angle) / steps
            pulse = self.calculate_servo_position(angle)
            self.set_servo_pulse(self.servo_channel, pulse)
            time.sleep(movement_speed)


# test_actuator_control.py
import unittest
from actuator_control import ActuatorControl

class TestActuatorControl(unittest.TestCase):

    def setUp(self):
        self.actuator = ActuatorControl()

    def test_calculate_servo_position(self):
        # Test para verificar que el cálculo de posición del servomotor es correcto
        angle = 90
        expected_pulse = int(self.actuator.servo_min + (self.actuator.servo_max - self.actuator.servo_min) * (angle / 180.0))
        result_pulse = self.actuator.calculate_servo_position(angle)
        self.assertEqual(result_pulse, expected_pulse, f"Expected {expected_pulse}, but got {result_pulse}")

    def test_set_servo_pulse(self):
        # Test para verificar que el pulso se configura correctamente (requiere hardware o mock)
        # Aquí simplemente verificamos que la función no lanza excepciones
        try:
            self.actuator.set_servo_pulse(self.actuator.servo_channel, 300)
        except Exception as e:
            self.fail(f"set_servo_pulse raised an exception: {str(e)}")

    def test_move_servo_smoothly(self):
        # Test para verificar que el servomotor se mueve suavemente (requiere hardware o mock)
        try:
            self.actuator.move_servo_smoothly(0, 90, 0.1)
        except Exception as e:
            self.fail(f"move_servo_smoothly raised an exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()



# actuator_control.py
from Adafruit_PCA9685 import PCA9685
import time

class ActuatorControl:
    def __init__(self):
        self.servo_channel = 0
        self.servo_min = 150
        self.servo_max = 600
        self.pwm = PCA9685()
        self.pwm.set_pwm_freq(50)
    
    def set_servo_pulse(self, channel, pulse):
        """Configura el pulso del servomotor en función del canal y el pulso dado."""
        self.pwm.set_pwm(channel, 0, pulse)
    
    def calculate_servo_position(self, angle):
        """Calcula el pulso del servomotor en función del ángulo deseado."""
        return int(self.servo_min + (self.servo_max - self.servo_min) * (angle / 180.0))
    
    def move_servo_smoothly(self, start_angle, end_angle, movement_speed):
        """Mueve el servomotor de forma suave entre dos ángulos."""
        start_pulse = self.calculate_servo_position(start_angle)
        end_pulse = self.calculate_servo_position(end_angle)
        steps = abs(end_angle - start_angle) // 1
        for step in range(steps + 1):
            angle = start_angle + step * (end_angle - start_angle) / steps
            pulse = self.calculate_servo_position(angle)
            self.set_servo_pulse(self.servo_channel, pulse)
            time.sleep(movement_speed)




# main_system.py
from flask import Flask, request, jsonify
import requests
from actuator_control import ActuatorControl

app = Flask(__name__)
actuator = ActuatorControl()

# Configuración de la URL del sistema de visión artificial
VISION_SYSTEM_URL = 'http://localhost:5001/detect'  # Cambia esto según la URL de tu sistema de visión

@app.route('/receive_data', methods=['POST'])
def receive_data():
    """Recibe datos del sistema de visión artificial y controla el actuador."""
    try:
        data = request.json
        cable_status = data.get('cable_status')
        if cable_status == 'dead':
            # Mover el actuador para cortar el cable
            actuator.move_servo_smoothly(0, 90, 0.1)
            return jsonify({"status": "success", "message": "Actuador movido para cortar el cable muerto"}), 200
        else:
            return jsonify({"status": "success", "message": "No se requiere acción"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/send_command', methods=['POST'])
def send_command():
    """Envía un comando al sistema de visión artificial."""
    try:
        command = request.json.get('command')
        response = requests.post(VISION_SYSTEM_URL, json={'command': command})
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




# vision_system.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    """Simula la detección de cables muertos y envía el estado al sistema principal."""
    try:
        command = request.json.get('command')
        # Aquí implementarías la lógica de detección real
        # Por simplicidad, simulamos la detección
        cable_status = 'dead' if command == 'cut_cable' else 'alive'
        return jsonify({"cable_status": cable_status}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Puerto diferente al del sistema principal
