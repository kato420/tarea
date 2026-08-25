#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <functional>
#include <vector>

class Tensor {
 private:
  std::vector<size_t> forma;
  double* datos;
  size_t tamano;

  static Tensor operacion_con_broadcasting (
      const Tensor& a, const Tensor& b,
      const std::function<double (double, double)>& op);

 public:
  Tensor ();
  Tensor (const std::vector<size_t>& shape, const std::vector<double>& values);
  ~Tensor ();
  Tensor (const Tensor& otro);
  Tensor& operator= (const Tensor& otro);
  Tensor (Tensor&& otro) noexcept;
  Tensor& operator= (Tensor&& otro) noexcept;

  static Tensor zeros (const std::vector<size_t>& shape);
  static Tensor ones (const std::vector<size_t>& shape);
  static Tensor random (const std::vector<size_t>& shape, double minimo,
                        double maximo);
  static Tensor arange (double inicio, double fin);

  Tensor operator+ (const Tensor& otro) const;
  Tensor operator- (const Tensor& otro) const;
  Tensor operator* (const Tensor& otro) const;
  Tensor operator* (double valor) const;

  Tensor view (const std::vector<size_t>& nueva_forma);
  Tensor unsqueeze (size_t dimension);
  static Tensor concat (const std::vector<Tensor>& tensores, size_t dimension);

  friend Tensor dot (const Tensor& a, const Tensor& b);
  friend Tensor matmul (const Tensor& a, const Tensor& b);

  const std::vector<size_t>& obtener_forma () const { return forma; }
  size_t obtener_tamano () const { return tamano; }
  const double* obtener_datos () const { return datos; }
};

#endif
