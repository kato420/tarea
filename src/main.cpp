#include "tensor.h"
#include <cmath>
#include <iomanip>
#include <iostream>

// Función de activación ReLU: pone en cero los valores negativos f(x) = max(0, x)
Tensor relu(const Tensor &t) {
  std::vector<double> res(t.obtener_tamano());
  const double *datos = t.obtener_datos();
  for (size_t i = 0; i < t.obtener_tamano(); ++i) {
    res[i] = datos[i] > 0.0 ? datos[i] : 0.0;
  }
  return Tensor(t.obtener_forma(), res);
}

// Función de activación Sigmoide: comprime los valores al rango (0, 1) mediante f(x) = 1 / (1 + exp(-x))
Tensor sigmoid(const Tensor &t) {
  std::vector<double> res(t.obtener_tamano());
  const double *datos = t.obtener_datos();
  for (size_t i = 0; i < t.obtener_tamano(); ++i) {
    res[i] = 1.0 / (1.0 + std::exp(-datos[i]));
  }
  return Tensor(t.obtener_forma(), res);
}

// Imprime la dimensión y tamaño total de elementos en cada paso
void imprimir_etapa(int paso, const std::string &operacion, const Tensor &t) {
  std::cout << "Paso " << paso << " | " << std::left << std::setw(28)
            << operacion << " | Dimensión: [";
  const auto &forma = t.obtener_forma();
  for (size_t i = 0; i < forma.size(); ++i) {
    std::cout << forma[i] << (i + 1 < forma.size() ? " x " : "");
  }
  std::cout << "] | Tamaño (size): " << t.obtener_tamano() << "\n";
}

int main() {
  std::cout << "======================================================================\n";
  std::cout << "       Flujo Completo de Red Neuronal utilizando Tensor++             \n";
  std::cout << "======================================================================\n\n";

  // Paso 1: Crear lote de entrada de 1000 imágenes de 20x20
  Tensor entrada = Tensor::random({1000, 20, 20}, -1.0, 1.0);
  imprimir_etapa(1, "Tensor entrada (datos crudos)", entrada);

  // Paso 2: Aplanar características a matriz 1000 x 400 usando view sin copiar memoria
  Tensor vista = entrada.view({1000, 400});
  imprimir_etapa(2, "view", vista);

  // Paso 3: Multiplicar por pesos W1 (400 x 100) acotados en [-0.1, 0.1] para no saturar
  Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
  Tensor capa1 = matmul(vista, W1);
  imprimir_etapa(3, "matmul con pesos W1", capa1);

  // Paso 4: Sumar vector de sesgo b1 (1 x 100) con broadcasting
  Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
  capa1 = capa1 + b1;
  imprimir_etapa(4, "Suma con bias b1 (1 x 100)", capa1);

  // Paso 5: Aplicar activación no lineal ReLU
  Tensor act1 = relu(capa1);
  imprimir_etapa(5, "Activación ReLU", act1);

  // Paso 6: Multiplicar por pesos W2 (100 x 10) para reducir a 10 salidas
  Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
  Tensor capa2 = matmul(act1, W2);
  imprimir_etapa(6, "matmul con pesos W2", capa2);

  // Paso 7: Sumar vector de sesgo b2 (1 x 10) con broadcasting
  Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
  capa2 = capa2 + b2;
  imprimir_etapa(7, "Suma con bias b2 (1 x 10)", capa2);

  // Paso 8: Aplicar activación final Sigmoide
  Tensor salida = sigmoid(capa2);
  imprimir_etapa(8, "Activación Sigmoid", salida);

  std::cout << "\n======================================================================\n";
  std::cout << "       Procesamiento de la red neuronal finalizado correctamente      \n";
  std::cout << "======================================================================\n";

  return 0;
}
