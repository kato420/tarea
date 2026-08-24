# Tensor++: Librería de Tensores en C++

**Tensor++** es una biblioteca científica en C++20 inspirada en NumPy y PyTorch para el manejo eficiente de tensores de hasta 3 dimensiones (1D, 2D y 3D). Implementa gestión de memoria dinámica en un arreglo contiguo (`double*`), semántica de movimiento para operaciones sin copias innecesarias, sobrecarga de operadores con soporte completo de *broadcasting*, concatenación multidimensional y transformaciones algebraicas aplicadas a una red neuronal profunda.

---

## 1. Estructura del Proyecto

```text
tarea/
├── CMakeLists.txt      # Configuración de compilación con CMake (C++20)
├── run                 # Script automatizado de compilación y ejecución
├── README.md           # Documentación técnica e instrucciones
└── src/
    ├── main.cpp        # Pipeline y ejecución de la red neuronal
    ├── tensor.h        # Declaración de la clase Tensor y funciones amigas
    └── tensor.cpp      # Implementación completa de Tensor++
```

---

## 2. Arquitectura y Gestión de Memoria

### 2.1 Almacenamiento Interno
Los datos del tensor se almacenan en un arreglo dinámico contiguo en memoria (`double *datos`), acompañado de:
* `std::vector<size_t> forma`: Dimensiones lógicas del tensor (máximo 3 dimensiones).
* `size_t tamano`: Número total de elementos ($N = \prod \text{forma}_i$).

### 2.2 Ciclo de Vida y Regla de los 5 (Rule of Five)
Para garantizar la integridad de la memoria y evitar fugas o doble liberación:
* **Constructor de Copia (`Tensor(const Tensor&)`):** Realiza una copia profunda (*deep copy*) de los datos dinámicos si el objeto origen contiene memoria asignada.
* **Asignador de Copia (`operator=(const Tensor&)`):** Protegido contra autoasignación; reserva nueva memoria antes de liberar la existente para asegurar seguridad ante excepciones.
* **Constructor de Movimiento (`Tensor(Tensor&&) noexcept`):** Transfiere la propiedad del puntero de datos en tiempo $O(1)$, dejando el tensor de origen en un estado válido pero vacío (`datos = nullptr`, `tamano = 0`).
* **Asignador de Movimiento (`operator=(Tensor&&) noexcept`):** Libera los recursos del objeto actual y toma posesión de los del temporal en $O(1)$.
* **Destructor (`~Tensor()`):** Libera la memoria dinámica asignada (`delete[] datos`).

---

## 3. Funcionalidades Implementadas

### 3.1 Métodos Estáticos de Creación
* `Tensor::zeros(shape)`: Inicializa todas las posiciones en `0.0`.
* `Tensor::ones(shape)`: Inicializa todas las posiciones en `1.0`.
* `Tensor::random(shape, min, max)`: Genera valores aleatorios uniformemente distribuidos en el rango $[\text{min}, \text{max})$ utilizando `std::mt19937` y `std::uniform_real_distribution`.
* `Tensor::arange(inicio, fin)`: Crea un tensor 1D con valores secuenciales $[\text{inicio}, \text{fin})$ con paso de $1.0$.

### 3.2 Sobrecarga de Operadores y Sistema de Broadcasting
Se sobrecargan los operadores aritméticos `+`, `-`, `*` (producto elemento a elemento de Hadamard) y `*` (multiplicación por escalar `double`).

#### Mecanismo de Broadcasting:
Cuando dos tensores tienen formas distintas pero compatibles, se aplica *broadcasting* multidimensional siguiendo el estándar de NumPy:
1. Las dimensiones se alinean a la derecha rellenando con $1$ a la izquierda hasta igualar el rango.
2. Dos dimensiones son compatibles si son iguales o si una de ellas es $1$.
3. Se calculan *strides* adaptativos donde las dimensiones unitarias tienen paso $0$, permitiendo operar directamente matrices $N \times M$ con vectores de sesgo $1 \times M$ sin replicación explícita de memoria.

### 3.3 Reorganización Dimensional (Zero-Copy)
* `view(nueva_forma)`: Reinterpreta la organización lógica de los datos sin duplicar memoria. Transfiere el búfer dinámico al nuevo tensor en $O(1)$ y deja el tensor original en un estado vacío seguro.
* `unsqueeze(dimension)`: Inserta una dimensión de tamaño $1$ en la posición indicada sin copiar datos, validando que el rango resultante no exceda 3 dimensiones.

### 3.4 Concatenación Multidimensional (`concat`)
El método estático `Tensor::concat(tensores, dimension)` une una lista de tensores a lo largo de un eje específico:
* Valida la compatibilidad dimensional estricta en todos los ejes distintos al de concatenación.
* Calcula la forma resultante sumando las dimensiones del eje elegido:
$$ \text{forma}_{\text{nueva}}[\text{dimension}] = \sum_{t} t.\text{forma}[\text{dimension}] $$
* Realiza una copia estructurada por bloques (*outer* e *inner slices*) que garantiza el orden contiguo correcto en memoria tanto para filas (eje 0), columnas (eje 1) o profundidad (eje 2).

### 3.5 Operaciones Algebraicas (Funciones Amigas)
* `friend Tensor dot(const Tensor& a, const Tensor& b)`: Calcula el producto punto entre dos tensores unidimensionales (1D) de igual tamaño: $\text{dot}(a, b) = \sum a_i \cdot b_i$.
* `friend Tensor matmul(const Tensor& a, const Tensor& b)`: Multiplica dos matrices bidimensionales (2D) compatibles ($A_{M \times K} \times B_{K \times N} \to C_{M \times N}$). Utiliza un orden de iteración $(i, k, j)$ para maximizar la localidad espacial en la memoria caché.

---

## 4. Aplicación Real: Red Neuronal Profunda

En `src/main.cpp` se implementa el flujo de procesamiento completo de una red neuronal feed-forward para procesar un lote de 1000 muestras de entrada.

### Flujo de Datos por Etapas

| Paso | Operación | Dimensión Resultante | Tamaño Total (`size`) |
| :---: | :--- | :---: | :---: |
| **1** | Tensor de entrada (datos crudos) | $1000 \times 20 \times 20$ | $400\,000$ |
| **2** | Aplanar características (`view`) | $1000 \times 400$ | $400\,000$ |
| **3** | `matmul` con pesos $W_1$ ($400 \times 100$) | $1000 \times 100$ | $100\,000$ |
| **4** | Suma con sesgo $b_1$ ($1 \times 100$) vía *broadcasting* | $1000 \times 100$ | $100\,000$ |
| **5** | Activación no lineal $\text{ReLU}(x) = \max(0, x)$ | $1000 \times 100$ | $100\,000$ |
| **6** | `matmul` con pesos $W_2$ ($100 \times 10$) | $1000 \times 10$ | $10\,000$ |
| **7** | Suma con sesgo $b_2$ ($1 \times 10$) vía *broadcasting* | $1000 \times 10$ | $10\,000$ |
| **8** | Activación final $\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$ | $1000 \times 10$ | $10\,000$ |

---

## 5. Instrucciones de Compilación y Ejecución

### 5.1 Requisitos del Sistema
* Compilador de C++ con soporte para **C++20** (`g++ >= 11` o `clang++ >= 13`).
* **CMake** versión 3.20 o superior.
* Generador **Ninja** (recomendado) o **Make**.

### 5.2 Opción 1: Ejecución Rápida (Script)
El repositorio incluye el script `run` para compilar incrementalmente y ejecutar en un solo paso:
```bash
./run
```

### 5.3 Opción 2: Compilación Manual con CMake
```bash
# 1. Crear y configurar el directorio de construcción
cmake -S . -B build -G Ninja

# 2. Compilar el proyecto
cmake --build build

# 3. Ejecutar el binario generado
./build/tarea
```

### 5.4 Opción 3: Compilación Directa con g++
```bash
g++ -std=c++20 -O3 src/main.cpp src/tensor.cpp -o build/tarea
./build/tarea
```
