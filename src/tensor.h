#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

class Tensor {
 private:
  std::vector<size_t> forma;  // Guarda las dimensiones del tensor (1D, 2D o 3D)
  double*
      datos;  // Puntero al arreglo contiguo en memoria donde van los números
  size_t tamano;  // Cantidad total de elementos almacenados (producto de las
                  // dimensiones)

  // Función auxiliar con plantilla para hacer broadcasting e inlinear la
  // operación aritmética
  template <typename Op>
  static Tensor operacion_con_broadcasting (const Tensor& a, const Tensor& b,
                                            Op op);

 public:
  // Constructor por defecto: crea un tensor vacío sin memoria asignada
  Tensor ();

  // Constructor principal: reserva memoria contigua y copia los valores
  // recibidos
  Tensor (const std::vector<size_t>& shape, const std::vector<double>& values);

  // Destructor: libera la memoria dinámica del arreglo de datos
  ~Tensor ();

  // Constructor de copia: crea un nuevo tensor con su propio bloque de memoria
  // (copia profunda)
  Tensor (const Tensor& otro);

  // Operador de asignación por copia: libera la memoria actual y copia los
  // datos del otro tensor
  Tensor& operator= (const Tensor& otro);

  // Constructor de movimiento: transfiere el puntero del otro tensor y lo deja
  // en null en O(1)
  Tensor (Tensor&& otro) noexcept;

  // Asignación por movimiento: toma posesión del puntero temporal y libera los
  // recursos viejos
  Tensor& operator= (Tensor&& otro) noexcept;

  // Métodos estáticos para crear tensores inicializados
  static Tensor zeros (
      const std::vector<size_t>& shape);  // Llena el tensor con ceros 0.0
  static Tensor ones (
      const std::vector<size_t>& shape);  // Llena el tensor con unos 1.0
  static Tensor random (const std::vector<size_t>& shape, double minimo,
                        double maximo);  // Valores aleatorios uniformes
  static Tensor arange (double inicio,
                        double fin);  // Vector 1D con valores secuenciales

  // Acceso directo a elementos por índices (sobrecarga de paréntesis)
  double& operator() (size_t i);  // Acceso 1D
  const double& operator() (size_t i) const;
  double& operator() (size_t i, size_t j);  // Acceso 2D (fila, columna)
  const double& operator() (size_t i, size_t j) const;
  double& operator() (size_t i, size_t j, size_t k);  // Acceso 3D
  const double& operator() (size_t i, size_t j, size_t k) const;

  // Acceso seguro con validación de límites lanzando std::out_of_range
  double& at (const std::vector<size_t>& indices);
  const double& at (const std::vector<size_t>& indices) const;

  // Operaciones aritméticas elemento a elemento con broadcasting
  Tensor operator+ (const Tensor& otro) const;
  Tensor operator- (const Tensor& otro) const;
  Tensor operator* (const Tensor& otro) const;  // Producto de Hadamard
  Tensor operator* (double valor) const;        // Multiplicación por escalar

  // Modificación de forma y concatenación
  Tensor view (const std::vector<size_t>&
                   nueva_forma);  // Cambia la forma sin copiar memoria
  Tensor unsqueeze (
      size_t dimension);  // Inserta dimensión 1 en la posición dada
  static Tensor concat (const std::vector<Tensor>& tensores,
                        size_t dimension);  // Une tensores a lo largo de un eje

  // Funciones amigas para operaciones algebraicas e impresión
  friend Tensor dot (const Tensor& a,
                     const Tensor& b);  // Producto punto entre vectores 1D
  friend Tensor matmul (
      const Tensor& a,
      const Tensor& b);  // Multiplicación matricial 2D (M x K) * (K x N)
  friend std::ostream& operator<< (
      std::ostream& os,
      const Tensor& t);  // Imprime el tensor formateado en consola

  // Métodos getter para inspeccionar el tensor
  const std::vector<size_t>& obtener_forma () const { return forma; }
  size_t obtener_tamano () const { return tamano; }
  const double* obtener_datos () const { return datos; }
};

#endif
