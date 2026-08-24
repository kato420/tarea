#include "tensor.h"
#include <algorithm>
#include <cmath>
#include <random>

Tensor::Tensor() : datos(nullptr), tamano(0) {}

Tensor::Tensor(const std::vector<size_t> &shape,
               const std::vector<double> &values) {
  if (shape.empty() || shape.size() > 3) {
    throw std::invalid_argument(
        "El tensor debe tener entre 1 y 3 dimensiones.");
  }
  tamano = 1;
  for (size_t dim : shape) {
    if (dim == 0) {
      throw std::invalid_argument("Las dimensiones deben ser mayores a cero.");
    }
    tamano *= dim;
  }
  if (tamano != values.size()) {
    throw std::invalid_argument("La cantidad de valores no coincide con el "
                                "producto de las dimensiones.");
  }
  forma = shape;
  datos = new double[tamano];
  std::copy(values.begin(), values.end(), datos);
}

Tensor::~Tensor() {
  delete[] datos;
  datos = nullptr;
  tamano = 0;
}

Tensor::Tensor(const Tensor &otro) : forma(otro.forma), tamano(otro.tamano) {
  if (otro.datos && tamano > 0) {
    datos = new double[tamano];
    std::copy(otro.datos, otro.datos + tamano, datos);
  } else {
    datos = nullptr;
    tamano = 0;
  }
}

Tensor &Tensor::operator=(const Tensor &otro) {
  if (this != &otro) {
    double *nuevos_datos = nullptr;
    if (otro.datos && otro.tamano > 0) {
      nuevos_datos = new double[otro.tamano];
      std::copy(otro.datos, otro.datos + otro.tamano, nuevos_datos);
    }
    delete[] datos;
    datos = nuevos_datos;
    forma = otro.forma;
    tamano = otro.tamano;
  }
  return *this;
}

Tensor::Tensor(Tensor &&otro) noexcept
    : forma(std::move(otro.forma)), datos(otro.datos), tamano(otro.tamano) {
  otro.datos = nullptr;
  otro.tamano = 0;
}

Tensor &Tensor::operator=(Tensor &&otro) noexcept {
  if (this != &otro) {
    delete[] datos;
    forma = std::move(otro.forma);
    datos = otro.datos;
    tamano = otro.tamano;
    otro.datos = nullptr;
    otro.tamano = 0;
  }
  return *this;
}

Tensor Tensor::zeros(const std::vector<size_t> &shape) {
  if (shape.empty() || shape.size() > 3) {
    throw std::invalid_argument(
        "El tensor debe tener entre 1 y 3 dimensiones.");
  }
  size_t total = 1;
  for (size_t dim : shape) {
    if (dim == 0)
      throw std::invalid_argument("Las dimensiones deben ser mayores a cero.");
    total *= dim;
  }
  return Tensor(shape, std::vector<double>(total, 0.0));
}

Tensor Tensor::ones(const std::vector<size_t> &shape) {
  if (shape.empty() || shape.size() > 3) {
    throw std::invalid_argument(
        "El tensor debe tener entre 1 y 3 dimensiones.");
  }
  size_t total = 1;
  for (size_t dim : shape) {
    if (dim == 0)
      throw std::invalid_argument("Las dimensiones deben ser mayores a cero.");
    total *= dim;
  }
  return Tensor(shape, std::vector<double>(total, 1.0));
}

Tensor Tensor::random(const std::vector<size_t> &shape, double minimo,
                      double maximo) {
  if (shape.empty() || shape.size() > 3) {
    throw std::invalid_argument(
        "El tensor debe tener entre 1 y 3 dimensiones.");
  }
  size_t total = 1;
  for (size_t dim : shape) {
    if (dim == 0)
      throw std::invalid_argument("Las dimensiones deben ser mayores a cero.");
    total *= dim;
  }
  std::vector<double> valores(total);
  std::random_device generador;
  std::mt19937 motor(generador());
  std::uniform_real_distribution<double> dist(minimo, maximo);
  for (size_t i = 0; i < total; ++i) {
    valores[i] = dist(motor);
  }
  return Tensor(shape, valores);
}

Tensor Tensor::arange(double inicio, double fin) {
  if (inicio >= fin) {
    throw std::invalid_argument(
        "El valor de inicio debe ser menor al de fin en arange.");
  }
  std::vector<double> valores;
  for (double i = inicio; i < fin; i += 1.0) {
    valores.push_back(i);
  }
  return Tensor({valores.size()}, valores);
}

Tensor Tensor::operacion_con_broadcasting(
    const Tensor &a, const Tensor &b,
    const std::function<double(double, double)> &op) {
  if (a.forma.empty() || b.forma.empty() || !a.datos || !b.datos) {
    throw std::invalid_argument("No se pueden operar tensores vacíos.");
  }

  if (a.forma == b.forma) {
    std::vector<double> res(a.tamano);
    for (size_t i = 0; i < a.tamano; ++i) {
      res[i] = op(a.datos[i], b.datos[i]);
    }
    return Tensor(a.forma, res);
  }

  size_t rank = std::max(a.forma.size(), b.forma.size());
  if (rank > 3) {
    throw std::invalid_argument(
        "El número máximo de dimensiones soportado es 3.");
  }

  std::vector<size_t> sa(rank, 1);
  std::vector<size_t> sb(rank, 1);
  for (size_t i = 0; i < a.forma.size(); ++i) {
    sa[rank - a.forma.size() + i] = a.forma[i];
  }
  for (size_t i = 0; i < b.forma.size(); ++i) {
    sb[rank - b.forma.size() + i] = b.forma[i];
  }

  std::vector<size_t> forma_salida(rank);
  size_t total_salida = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (sa[i] == sb[i]) {
      forma_salida[i] = sa[i];
    } else if (sa[i] == 1) {
      forma_salida[i] = sb[i];
    } else if (sb[i] == 1) {
      forma_salida[i] = sa[i];
    } else {
      throw std::invalid_argument(
          "Dimensiones incompatibles para broadcasting.");
    }
    total_salida *= forma_salida[i];
  }

  std::vector<size_t> strides_a(rank, 0);
  std::vector<size_t> strides_b(rank, 0);

  size_t cur_stride_a = 1;
  size_t cur_stride_b = 1;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    if (sa[i] > 1) {
      strides_a[i] = cur_stride_a;
      cur_stride_a *= sa[i];
    }
    if (sb[i] > 1) {
      strides_b[i] = cur_stride_b;
      cur_stride_b *= sb[i];
    }
  }

  std::vector<double> res(total_salida);
  std::vector<size_t> coords(rank, 0);

  for (size_t idx = 0; idx < total_salida; ++idx) {
    size_t off_a = 0;
    size_t off_b = 0;
    for (size_t r = 0; r < rank; ++r) {
      off_a += coords[r] * strides_a[r];
      off_b += coords[r] * strides_b[r];
    }

    res[idx] = op(a.datos[off_a], b.datos[off_b]);

    for (int r = static_cast<int>(rank) - 1; r >= 0; --r) {
      coords[r]++;
      if (coords[r] < forma_salida[r]) {
        break;
      }
      coords[r] = 0;
    }
  }

  return Tensor(forma_salida, res);
}

Tensor Tensor::operator+(const Tensor &otro) const {
  return operacion_con_broadcasting(*this, otro,
                                    [](double x, double y) { return x + y; });
}

Tensor Tensor::operator-(const Tensor &otro) const {
  return operacion_con_broadcasting(*this, otro,
                                    [](double x, double y) { return x - y; });
}

Tensor Tensor::operator*(const Tensor &otro) const {
  return operacion_con_broadcasting(*this, otro,
                                    [](double x, double y) { return x * y; });
}

Tensor Tensor::operator*(double valor) const {
  if (tamano == 0 || !datos) {
    return Tensor();
  }
  std::vector<double> res(tamano);
  for (size_t i = 0; i < tamano; ++i) {
    res[i] = datos[i] * valor;
  }
  return Tensor(forma, res);
}

Tensor Tensor::view(const std::vector<size_t> &nueva_forma) {
  if (nueva_forma.empty() || nueva_forma.size() > 3) {
    throw std::invalid_argument(
        "La nueva forma debe tener entre 1 y 3 dimensiones.");
  }
  size_t nuevo_tamano = 1;
  for (size_t dim : nueva_forma) {
    if (dim == 0) {
      throw std::invalid_argument("Las dimensiones deben ser mayores a cero.");
    }
    nuevo_tamano *= dim;
  }
  if (nuevo_tamano != tamano) {
    throw std::invalid_argument(
        "El número total de elementos debe coincidir para la operación view.");
  }

  Tensor resultado;
  resultado.forma = nueva_forma;
  resultado.tamano = tamano;
  resultado.datos = datos;

  datos = nullptr;
  tamano = 0;
  forma.clear();

  return resultado;
}

Tensor Tensor::unsqueeze(size_t dimension) {
  if (forma.size() >= 3) {
    throw std::invalid_argument(
        "No se puede hacer unsqueeze: el tensor ya tiene 3 dimensiones.");
  }
  if (dimension > forma.size()) {
    throw std::invalid_argument(
        "La dimensión indicada para unsqueeze está fuera de rango.");
  }

  std::vector<size_t> nueva_forma = forma;
  nueva_forma.insert(nueva_forma.begin() + dimension, 1);

  Tensor resultado;
  resultado.forma = nueva_forma;
  resultado.tamano = tamano;
  resultado.datos = datos;

  datos = nullptr;
  tamano = 0;
  forma.clear();

  return resultado;
}

Tensor Tensor::concat(const std::vector<Tensor> &tensores, size_t dimension) {
  if (tensores.empty()) {
    throw std::invalid_argument(
        "La lista de tensores para concatenar no puede estar vacía.");
  }

  size_t rank = tensores[0].forma.size();
  if (rank == 0 || rank > 3) {
    throw std::invalid_argument(
        "Número de dimensiones inválido para concatenar.");
  }
  if (dimension >= rank) {
    throw std::invalid_argument(
        "La dimensión de concatenación excede el rango del tensor.");
  }

  size_t total_dim_concat = 0;
  for (const auto &t : tensores) {
    if (t.forma.size() != rank) {
      throw std::invalid_argument("Todos los tensores a concatenar deben tener "
                                  "el mismo número de dimensiones.");
    }
    for (size_t d = 0; d < rank; ++d) {
      if (d != dimension && t.forma[d] != tensores[0].forma[d]) {
        throw std::invalid_argument(
            "Las dimensiones no concatenadas deben coincidir exactamente.");
      }
    }
    total_dim_concat += t.forma[dimension];
  }

  std::vector<size_t> nueva_forma = tensores[0].forma;
  nueva_forma[dimension] = total_dim_concat;

  size_t total_elementos = 1;
  for (size_t d : nueva_forma) {
    total_elementos *= d;
  }

  std::vector<double> unidos(total_elementos);

  size_t outer_count = 1;
  for (size_t d = 0; d < dimension; ++d) {
    outer_count *= nueva_forma[d];
  }

  size_t inner_count = 1;
  for (size_t d = dimension + 1; d < rank; ++d) {
    inner_count *= nueva_forma[d];
  }

  for (size_t o = 0; o < outer_count; ++o) {
    size_t offset_in_dim = 0;
    for (const auto &t : tensores) {
      size_t dim_t = t.forma[dimension];
      for (size_t d = 0; d < dim_t; ++d) {
        const double *src = t.datos + (o * dim_t + d) * inner_count;
        double *dest =
            unidos.data() +
            (o * total_dim_concat + (offset_in_dim + d)) * inner_count;
        std::copy(src, src + inner_count, dest);
      }
      offset_in_dim += dim_t;
    }
  }

  return Tensor(nueva_forma, unidos);
}

Tensor dot(const Tensor &a, const Tensor &b) {
  if (a.forma.size() != 1 || b.forma.size() != 1 || a.tamano != b.tamano) {
    throw std::invalid_argument(
        "dot requiere tensores unidimensionales (1D) de igual tamaño.");
  }
  double suma = 0.0;
  for (size_t i = 0; i < a.tamano; ++i) {
    suma += a.datos[i] * b.datos[i];
  }
  return Tensor({1}, {suma});
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.forma.size() != 2 || b.forma.size() != 2 || a.forma[1] != b.forma[0]) {
    throw std::invalid_argument(
        "Dimensiones incompatibles para matmul (se requiere A[M,K] y B[K,N]).");
  }
  size_t filas = a.forma[0];
  size_t comunes = a.forma[1];
  size_t columnas = b.forma[1];

  std::vector<double> res(filas * columnas, 0.0);
  for (size_t i = 0; i < filas; ++i) {
    for (size_t k = 0; k < comunes; ++k) {
      double a_ik = a.datos[i * comunes + k];
      for (size_t j = 0; j < columnas; ++j) {
        res[i * columnas + j] += a_ik * b.datos[k * columnas + j];
      }
    }
  }
  return Tensor({filas, columnas}, res);
}
