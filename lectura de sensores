// Configura los pines de los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configura las constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float voltajeOffset = 2.5; // Offset del sensor de voltaje (en voltios, para el divisor de voltaje)
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas
float voltaje;
float corriente;

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
}

// Bucle principal del programa
void loop() {
  // Leer los valores de los sensores
  int lecturaVoltaje = analogRead(voltajePin);
  int lecturaCorriente = analogRead(corrientePin);

  // Convertir las lecturas analógicas a voltaje
  voltaje = (lecturaVoltaje * voltajeReferencia) / 1024.0;
  corriente = ((lecturaCorriente * voltajeReferencia) / 1024.0 - voltajeOffset) / corrienteSensibilidad;

  // Mostrar los resultados en el monitor serial
  Serial.print("Voltaje: ");
  Serial.print(voltaje);
  Serial.print(" V, Corriente: ");
  Serial.print(corriente);
  Serial.println(" A");

  // Esperar antes de la siguiente lectura
  delay(1000); // Espera 1 segundo
}


// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas y calibración
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Calibración del sensor de corriente
  calibrarSensorCorriente();
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delay(10); // Espera para evitar lecturas demasiado rápidas
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Bucle principal del programa
void loop() {
  // Leer los valores de los sensores
  int lecturaVoltaje = analogRead(voltajePin);
  int lecturaCorriente = analogRead(corrientePin);

  // Convertir las lecturas analógicas a voltaje
  voltaje = (lecturaVoltaje * voltajeReferencia) / 1024.0;
  
  // Convertir la lectura de corriente a amperios, aplicando el offset calibrado
  corriente = ((lecturaCorriente * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;

  // Mostrar los resultados en el monitor serial
  Serial.print("Voltaje: ");
  Serial.print(voltaje);
  Serial.print(" V, Corriente: ");
  Serial.print(corriente);
  Serial.println(" A");

  // Esperar antes de la siguiente lectura
  delay(1000); // Espera 1 segundo
}


// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)
const float voltajeOffset = 0.0; // Offset para el divisor de voltaje (ajustable si es necesario)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Calibración del sensor de corriente
  calibrarSensorCorriente();
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delay(10); // Espera para evitar lecturas demasiado rápidas
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  voltaje = leerVoltaje();
  corriente = leerCorriente();
  
  // Mostrar los resultados en el monitor serial
  Serial.print("Voltaje: ");
  Serial.print(voltaje);
  Serial.print(" V, Corriente: ");
  Serial.print(corriente);
  Serial.println(" A");
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Esperar antes de la siguiente lectura
  delay(1000); // Espera 1 segundo
}




// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)
const float voltajeOffset = 0.0; // Offset para el divisor de voltaje (ajustable si es necesario)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Calibración del sensor de corriente
  calibrarSensorCorriente();
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delay(10); // Espera para evitar lecturas demasiado rápidas
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  
  // Manejo de errores en la lectura de corriente
  if (lecturaCorriente < 0 || lecturaCorriente > 1023) {
    Serial.println("Error: Lectura de corriente fuera de rango.");
    return -1; // Valor de error
  }

  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  
  // Manejo de errores en la lectura de voltaje
  if (lecturaVoltaje < 0 || lecturaVoltaje > 1023) {
    Serial.println("Error: Lectura de voltaje fuera de rango.");
    return -1; // Valor de error
  }

  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  voltaje = leerVoltaje();
  corriente = leerCorriente();
  
  // Verificar si los valores de voltaje y corriente son válidos
  if (voltaje >= 0 && corriente >= 0) {
    // Mostrar los resultados en el monitor serial
    Serial.print("Voltaje: ");
    Serial.print(voltaje);
    Serial.print(" V, Corriente: ");
    Serial.print(corriente);
    Serial.println(" A");
  } else {
    Serial.println("Error: Datos no válidos.");
  }
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Esperar antes de la siguiente lectura
  delay(1000); // Espera 1 segundo
}





// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración del filtro de media móvil
const int tamanoVentana = 10; // Número de muestras para el filtro
float corrienteMuestras[tamanoVentana];
float voltajeMuestras[tamanoVentana];
int indiceMuestra = 0;

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Inicializar las muestras con 0
  for (int i = 0; i < tamanoVentana; i++) {
    corrienteMuestras[i] = 0.0;
    voltajeMuestras[i] = 0.0;
  }

  // Calibración del sensor de corriente
  calibrarSensorCorriente();
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delay(10); // Espera para evitar lecturas demasiado rápidas
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  
  // Manejo de errores en la lectura de corriente
  if (lecturaCorriente < 0 || lecturaCorriente > 1023) {
    Serial.println("Error: Lectura de corriente fuera de rango.");
    return -1; // Valor de error
  }

  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  
  // Manejo de errores en la lectura de voltaje
  if (lecturaVoltaje < 0 || lecturaVoltaje > 1023) {
    Serial.println("Error: Lectura de voltaje fuera de rango.");
    return -1; // Valor de error
  }

  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para aplicar el filtro de media móvil
float aplicarFiltroMediaMovil(float muestra[], float nuevoValor) {
  // Eliminar el valor más antiguo y añadir el nuevo valor
  muestra[indiceMuestra] = nuevoValor;
  indiceMuestra = (indiceMuestra + 1) % tamanoVentana;
  
  // Calcular el promedio de las muestras
  float suma = 0.0;
  for (int i = 0; i < tamanoVentana; i++) {
    suma += muestra[i];
  }
  return suma / tamanoVentana;
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  float lecturaVoltaje = leerVoltaje();
  float lecturaCorriente = leerCorriente();
  
  if (lecturaVoltaje >= 0 && lecturaCorriente >= 0) {
    // Aplicar el filtro de media móvil
    voltaje = aplicarFiltroMediaMovil(voltajeMuestras, lecturaVoltaje);
    corriente = aplicarFiltroMediaMovil(corrienteMuestras, lecturaCorriente);
    
    // Mostrar los resultados en el monitor serial
    Serial.print("Voltaje (filtrado): ");
    Serial.print(voltaje);
    Serial.print(" V, Corriente (filtrada): ");
    Serial.print(corriente);
    Serial.println(" A");
  } else {
    Serial.println("Error: Datos no válidos.");
  }
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Esperar antes de la siguiente lectura
  delay(1000); // Espera 1 segundo
}




// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración del filtro de media móvil
const int tamanoVentana = 10; // Número de muestras para el filtro
float corrienteMuestras[tamanoVentana];
float voltajeMuestras[tamanoVentana];
int indiceMuestra = 0;

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Inicializar las muestras con 0
  for (int i = 0; i < tamanoVentana; i++) {
    corrienteMuestras[i] = 0.0;
    voltajeMuestras[i] = 0.0;
  }

  // Calibración del sensor de corriente
  calibrarSensorCorriente();
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delayMicroseconds(100); // Reducción del retardo a microsegundos
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  
  // Manejo de errores en la lectura de corriente
  if (lecturaCorriente < 0 || lecturaCorriente > 1023) {
    Serial.println("Error: Lectura de corriente fuera de rango.");
    return -1; // Valor de error
  }

  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  
  // Manejo de errores en la lectura de voltaje
  if (lecturaVoltaje < 0 || lecturaVoltaje > 1023) {
    Serial.println("Error: Lectura de voltaje fuera de rango.");
    return -1; // Valor de error
  }

  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para aplicar el filtro de media móvil
float aplicarFiltroMediaMovil(float muestra[], float nuevoValor) {
  // Eliminar el valor más antiguo y añadir el nuevo valor
  muestra[indiceMuestra] = nuevoValor;
  indiceMuestra = (indiceMuestra + 1) % tamanoVentana;
  
  // Calcular el promedio de las muestras
  float suma = 0.0;
  for (int i = 0; i < tamanoVentana; i++) {
    suma += muestra[i];
  }
  return suma / tamanoVentana;
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  // Leer los valores de los sensores
  float lecturaVoltaje = leerVoltaje();
  float lecturaCorriente = leerCorriente();
  
  if (lecturaVoltaje >= 0 && lecturaCorriente >= 0) {
    // Aplicar el filtro de media móvil
    voltaje = aplicarFiltroMediaMovil(voltajeMuestras, lecturaVoltaje);
    corriente = aplicarFiltroMediaMovil(corrienteMuestras, lecturaCorriente);
    
    // Mostrar los resultados en el monitor serial
    Serial.print("Voltaje (filtrado): ");
    Serial.print(voltaje);
    Serial.print(" V, Corriente (filtrada): ");
    Serial.print(corriente);
    Serial.println(" A");
  } else {
    Serial.println("Error: Datos no válidos.");
  }
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Utilizar un temporizador para reducir la frecuencia de las lecturas
  delay(500); // Espera 500 milisegundos
}


#include <EEPROM.h> // Incluye la librería EEPROM para almacenamiento en memoria

// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración del filtro de media móvil
const int tamanoVentana = 10; // Número de muestras para el filtro
float corrienteMuestras[tamanoVentana];
float voltajeMuestras[tamanoVentana];
int indiceMuestra = 0;

// Direcciones de memoria EEPROM para almacenamiento de datos
const int direccionVoltaje = 0; // Dirección de memoria EEPROM para el voltaje
const int direccionCorriente = sizeof(float); // Dirección de memoria EEPROM para la corriente

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Inicializar las muestras con 0
  for (int i = 0; i < tamanoVentana; i++) {
    corrienteMuestras[i] = 0.0;
    voltajeMuestras[i] = 0.0;
  }

  // Calibración del sensor de corriente
  calibrarSensorCorriente();
  
  // Leer datos almacenados desde EEPROM
  float voltajeAlmacenado;
  float corrienteAlmacenada;
  EEPROM.get(direccionVoltaje, voltajeAlmacenado);
  EEPROM.get(direccionCorriente, corrienteAlmacenada);
  
  Serial.print("Voltaje almacenado: ");
  Serial.println(voltajeAlmacenado);
  Serial.print("Corriente almacenada: ");
  Serial.println(corrienteAlmacenada);
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delayMicroseconds(100); // Reducción del retardo a microsegundos
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  
  // Manejo de errores en la lectura de corriente
  if (lecturaCorriente < 0 || lecturaCorriente > 1023) {
    Serial.println("Error: Lectura de corriente fuera de rango.");
    return -1; // Valor de error
  }

  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  
  // Manejo de errores en la lectura de voltaje
  if (lecturaVoltaje < 0 || lecturaVoltaje > 1023) {
    Serial.println("Error: Lectura de voltaje fuera de rango.");
    return -1; // Valor de error
  }

  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para aplicar el filtro de media móvil
float aplicarFiltroMediaMovil(float muestra[], float nuevoValor) {
  // Eliminar el valor más antiguo y añadir el nuevo valor
  muestra[indiceMuestra] = nuevoValor;
  indiceMuestra = (indiceMuestra + 1) % tamanoVentana;
  
  // Calcular el promedio de las muestras
  float suma = 0.0;
  for (int i = 0; i < tamanoVentana; i++) {
    suma += muestra[i];
  }
  return suma / tamanoVentana;
}

// Función para almacenar datos en EEPROM
void almacenarDatosEEPROM(float voltaje, float corriente) {
  EEPROM.put(direccionVoltaje, voltaje);
  EEPROM.put(direccionCorriente, corriente);
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  float lecturaVoltaje = leerVoltaje();
  float lecturaCorriente = leerCorriente();
  
  if (lecturaVoltaje >= 0 && lecturaCorriente >= 0) {
    // Aplicar el filtro de media móvil
    voltaje = aplicarFiltroMediaMovil(voltajeMuestras, lecturaVoltaje);
    corriente = aplicarFiltroMediaMovil(corrienteMuestras, lecturaCorriente);
    
    // Mostrar los resultados en el monitor serial
    Serial.print("Voltaje (filtrado): ");
    Serial.print(voltaje);
    Serial.print(" V, Corriente (filtrada): ");
    Serial.print(corriente);
    Serial.println(" A");
    
    // Almacenar los datos en EEPROM
    almacenarDatosEEPROM(voltaje, corriente);
  } else {
    Serial.println("Error: Datos no válidos.");
  }
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Utilizar un temporizador para reducir la frecuencia de las lecturas
  delay(1000); // Espera 1 segundo
}



#include <EEPROM.h> // Incluye la librería EEPROM para almacenamiento en memoria

// Declaración de pines para los sensores
const int corrientePin = A0; // Pin analógico para el sensor de corriente
const int voltajePin = A1;  // Pin analógico para el sensor de voltaje

// Configuración de constantes para los sensores
const float voltajeReferencia = 5.0; // Voltaje de referencia del Arduino
const float corrienteSensibilidad = 0.185; // Sensibilidad del sensor ACS712 (mV/A)

// Variables para almacenar las lecturas
float voltaje;
float corriente;
float corrienteOffset = 0.0; // Valor inicial para el offset de corriente, ajustable durante la calibración

// Configuración del filtro de media móvil
const int tamanoVentana = 10; // Número de muestras para el filtro
float corrienteMuestras[tamanoVentana];
float voltajeMuestras[tamanoVentana];
int indiceMuestra = 0;

// Direcciones de memoria EEPROM para almacenamiento de datos
const int direccionVoltaje = 0; // Dirección de memoria EEPROM para el voltaje
const int direccionCorriente = sizeof(float); // Dirección de memoria EEPROM para la corriente

// Configuración inicial del Arduino
void setup() {
  Serial.begin(9600); // Inicializa la comunicación serial a 9600 baudios
  
  // Inicializar las muestras con 0
  for (int i = 0; i < tamanoVentana; i++) {
    corrienteMuestras[i] = 0.0;
    voltajeMuestras[i] = 0.0;
  }

  // Calibración del sensor de corriente
  calibrarSensorCorriente();
  
  // Leer datos almacenados desde EEPROM
  float voltajeAlmacenado;
  float corrienteAlmacenada;
  EEPROM.get(direccionVoltaje, voltajeAlmacenado);
  EEPROM.get(direccionCorriente, corrienteAlmacenada);
  
  Serial.print("Voltaje almacenado: ");
  Serial.println(voltajeAlmacenado);
  Serial.print("Corriente almacenada: ");
  Serial.println(corrienteAlmacenada);
}

// Función para calibrar el sensor de corriente
void calibrarSensorCorriente() {
  Serial.println("Calibrando sensor de corriente...");
  long sumaLecturas = 0;
  const int numLecturas = 100;

  // Leer varias veces el valor del sensor para encontrar el offset
  for (int i = 0; i < numLecturas; i++) {
    sumaLecturas += analogRead(corrientePin);
    delayMicroseconds(100); // Reducción del retardo a microsegundos
  }
  
  // Calcular el offset promedio
  corrienteOffset = (sumaLecturas / numLecturas) * voltajeReferencia / 1024.0;
  Serial.print("Offset de corriente calibrado: ");
  Serial.println(corrienteOffset);
}

// Función para leer la corriente
float leerCorriente() {
  int lecturaCorriente = analogRead(corrientePin);
  
  // Manejo de errores en la lectura de corriente
  if (lecturaCorriente < 0 || lecturaCorriente > 1023) {
    Serial.println("Error: Lectura de corriente fuera de rango.");
    return -1; // Valor de error
  }

  return convertirCorriente(lecturaCorriente);
}

// Función para leer el voltaje
float leerVoltaje() {
  int lecturaVoltaje = analogRead(voltajePin);
  
  // Manejo de errores en la lectura de voltaje
  if (lecturaVoltaje < 0 || lecturaVoltaje > 1023) {
    Serial.println("Error: Lectura de voltaje fuera de rango.");
    return -1; // Valor de error
  }

  return convertirVoltaje(lecturaVoltaje);
}

// Función para convertir lecturas analógicas a valores de corriente
float convertirCorriente(int lectura) {
  return ((lectura * voltajeReferencia) / 1024.0 - corrienteOffset) / corrienteSensibilidad;
}

// Función para convertir lecturas analógicas a valores de voltaje
float convertirVoltaje(int lectura) {
  return (lectura * voltajeReferencia) / 1024.0;
}

// Función para aplicar el filtro de media móvil
float aplicarFiltroMediaMovil(float muestra[], float nuevoValor) {
  // Eliminar el valor más antiguo y añadir el nuevo valor
  muestra[indiceMuestra] = nuevoValor;
  indiceMuestra = (indiceMuestra + 1) % tamanoVentana;
  
  // Calcular el promedio de las muestras
  float suma = 0.0;
  for (int i = 0; i < tamanoVentana; i++) {
    suma += muestra[i];
  }
  return suma / tamanoVentana;
}

// Función para almacenar datos en EEPROM
void almacenarDatosEEPROM(float voltaje, float corriente) {
  EEPROM.put(direccionVoltaje, voltaje);
  EEPROM.put(direccionCorriente, corriente);
}

// Función para leer y procesar datos de varios sensores
void leerYProcesarDatos() {
  float lecturaVoltaje = leerVoltaje();
  float lecturaCorriente = leerCorriente();
  
  if (lecturaVoltaje >= 0 && lecturaCorriente >= 0) {
    // Aplicar el filtro de media móvil
    voltaje = aplicarFiltroMediaMovil(voltajeMuestras, lecturaVoltaje);
    corriente = aplicarFiltroMediaMovil(corrienteMuestras, lecturaCorriente);
    
    // Mostrar los resultados en el monitor serial
    Serial.print("Voltaje (filtrado): ");
    Serial.print(voltaje);
    Serial.print(" V, Corriente (filtrada): ");
    Serial.print(corriente);
    Serial.println(" A");
    
    // Almacenar los datos en EEPROM
    almacenarDatosEEPROM(voltaje, corriente);
    
    // Enviar datos por serial
    enviarDatosSerial(voltaje, corriente);
  } else {
    Serial.println("Error: Datos no válidos.");
  }
}

// Función para enviar datos por serial
void enviarDatosSerial(float voltaje, float corriente) {
  Serial.print("Datos enviados - Voltaje: ");
  Serial.print(voltaje);
  Serial.print(" V, Corriente: ");
  Serial.print(corriente);
  Serial.println(" A");
}

// Bucle principal del programa
void loop() {
  leerYProcesarDatos();
  
  // Utilizar un temporizador para reducir la frecuencia de las lecturas
  delay(1000); // Espera 1 segundo
}
