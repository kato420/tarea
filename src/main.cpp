#include <cmath>
#include <iomanip>
#include <iostream>

#include "tensor.h"

Tensor relu (const Tensor& t) {
  std::vector<double> res (t.obtener_tamano ());
  const double* datos = t.obtener_datos ();
  for (size_t i = 0; i < t.obtener_tamano (); ++i) {
    res[i] = datos[i] > 0.0 ? datos[i] : 0.0;
  }
  return Tensor (t.obtener_forma (), res);
}

Tensor sigmoid (const Tensor& t) {
  std::vector<double> res (t.obtener_tamano ());
  const double* datos = t.obtener_datos ();
  for (size_t i = 0; i < t.obtener_tamano (); ++i) {
    res[i] = 1.0 / (1.0 + std::exp (-datos[i]));
  }
  return Tensor (t.obtener_forma (), res);
}

void imprimir_etapa (int paso, const std::string& operacion, const Tensor& t) {
  std::cout << "Paso " << paso << " | " << std::left << std::setw (28)
            << operacion << " | Dimensión: [";
  const auto& forma = t.obtener_forma ();
  for (size_t i = 0; i < forma.size (); ++i) {
    std::cout << forma[i] << (i + 1 < forma.size () ? " x " : "");
  }
  std::cout << "] | Tamaño (size): " << t.obtener_tamano () << "\n";
}

int main () {
  std::cout << "==============================================================="
               "=======\n\n";

  Tensor entrada = Tensor::random ({1000, 20, 20}, -1.0, 1.0);
  imprimir_etapa (1, "Tensor entrada (datos crudos)", entrada);

  Tensor vista = entrada.view ({1000, 400});
  imprimir_etapa (2, "view", vista);

  Tensor W1 = Tensor::random ({400, 100}, -1.0, 1.0);
  Tensor capa1 = matmul (vista, W1);
  imprimir_etapa (3, "matmul con pesos W1", capa1);

  Tensor b1 = Tensor::random ({1, 100}, -1.0, 1.0);
  capa1 = capa1 + b1;
  imprimir_etapa (4, "Suma con bias b1 (1 x 100)", capa1);

  Tensor act1 = relu (capa1);
  imprimir_etapa (5, "Activación ReLU", act1);

  Tensor W2 = Tensor::random ({100, 10}, -1.0, 1.0);
  Tensor capa2 = matmul (act1, W2);
  imprimir_etapa (6, "matmul con pesos W2", capa2);

  Tensor b2 = Tensor::random ({1, 10}, -1.0, 1.0);
  capa2 = capa2 + b2;
  imprimir_etapa (7, "Suma con bias b2 (1 x 10)", capa2);

  Tensor salida = sigmoid (capa2);
  imprimir_etapa (8, "Activación Sigmoid", salida);

  return 0;
}
