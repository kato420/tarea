#include <iostream>
#include <vector>
using namespace std;

class Tensor {
  double* unod;
  double** dosd;
  double*** tresd;
  int cont = 1;
  vector<size_t> vecShape;
  vector<double> vecValues;

 public:
  friend ostream& operator<< (ostream& os, const Tensor& t);
  Tensor (const std ::vector<size_t>& shape, const std ::vector<double>& values)
      : vecShape (shape), vecValues (values) {
    for (int i = 0; i < shape.size (); i++) {
      cont *= shape[i];
    }
    if (values.size () != cont) {
      throw runtime_error ("Cantidad de valores incorrectos");
    }
    if (shape.size () == 1) {
      unod = new double[values.size ()];
      for (int i = 0; i < values.size (); i++) {
        unod[i] = values[i];
      }
    } else if (shape.size () == 2) {
      dosd = new double*[shape[0]];
      for (int i = 0; i < shape[0]; i++) {
        dosd[i] = new double[shape[1]];
        for (int j = 0; j < shape[1]; j++) {
          dosd[i][j] = values[(i * shape[1]) + j];
        }
      }
    } else if (shape.size () == 3) {
      tresd = new double**[shape[0]];
      for (int i = 0; i < shape[0]; i++) {
        tresd[i] = new double*[shape[1]];
        for (int j = 0; j < shape[1]; j++) {
          tresd[i][j] = new double[shape[2]];
          for (int k = 0; k < shape[2]; k++) {
            tresd[i][j][k] =
                values[(i * shape[1] * shape[2]) + (j * shape[2]) + k];
          }
        }
      }
    }
  }
  Tensor (const Tensor& other)
      : cont (other.cont),
        vecShape (other.vecShape),
        vecValues (other.vecValues) {
    if (other.vecShape.size () == 1) {
      unod = new double[other.vecValues.size ()];
      this->unod = other.unod;
    } else if (other.vecShape.size () == 2) {
      this->dosd = other.dosd;
    } else if (other.vecShape.size () == 3) {
      this->tresd = other.tresd;
    }
    cout << "Constructor copia utilizado" << endl;
  }

  Tensor (Tensor&& other) noexcept {
    int dimension = other.vecShape.size ();
    cont = other.cont;
    other.cont = 0;
    vecShape = std::move (other.vecShape);
    vecValues = std::move (other.vecValues);

    if (dimension == 1) {
      unod = other.unod;
      other.unod = nullptr;
    } else if (dimension == 2) {
      dosd = other.dosd;
      other.dosd = nullptr;
    } else if (dimension == 3) {
      tresd = other.tresd;
      other.tresd = nullptr;
    }
  }
  Tensor& operator= (const Tensor& other) {
    int dimension = other.vecShape.size ();
    this->cont = 0;
    this->vecValues.clear ();
    this->vecShape.clear ();
    this->unod = nullptr;
    this->dosd = nullptr;
    this->tresd = nullptr;
    if (dimension == 1) {
      this->unod = other.unod;
    } else if (dimension == 2) {
      this->dosd = other.dosd;
    } else if (dimension == 3) {
      this->tresd = other.tresd;
    }
    this->cont = other.cont;
    this->vecShape = other.vecShape;
    this->vecValues = other.vecValues;
    return *this;
  }

  static Tensor zeros (const std ::vector<size_t>& shape) {
    int cont = 1;
    for (int i = 0; i < shape.size (); i++) {
      cont *= shape[i];
    }
    vector<double> values (cont, 0);
    Tensor tensor (shape, values);
    return tensor;
  }
  static Tensor ones (const std ::vector<size_t>& shape) {
    int cont = 1;
    for (int i = 0; i < shape.size (); i++) {
      cont *= shape[i];
    }
    vector<double> values (cont, 1);
    return Tensor (shape, values);
  }
  static Tensor random (const std ::vector<size_t>& shape, int min, int max) {
    int cont = 1;
    for (int i = 0; i < shape.size (); i++) {
      cont *= shape[i];
    }
    vector<double> values (cont);
    for (int i = 0; i < cont; i++) {
      values[i] = min + rand () % (max - min + 1);
      ;
    }

    return Tensor (shape, values);
  }
  static Tensor arrange (int start, int end) {
    vector<double> values (end - start);
    for (int j = start, i = 0; j < end; j++, i++) values[i] = j;
    vector<size_t> miVector;
    miVector.push_back (end - start);
    return Tensor (miVector, values);
    ;
  }

  ~Tensor () {
    if (this->vecShape.size () == 1) {
      delete[] unod;
    } else if (this->vecShape.size () == 2) {
      for (int i = 0; i < vecShape[0]; i++) {
        delete[] dosd[i];
      }
      delete[] dosd;
    } else if (this->vecShape.size () == 3) {
      for (int i = 0; i < vecShape[0]; i++) {
        for (int j = 0; j < vecShape[1]; j++) {
          delete[] tresd[i][j];
        }
        delete[] tresd[i];
      }
      delete[] tresd;
    }
  }
};
ostream& operator<< (ostream& os, const Tensor& t) {
  if (t.vecShape.size () == 1) {
    for (int i = 0; i < t.cont; i++) {
      os << t.unod[i] << " ";
    }
  } else if (t.vecShape.size () == 2) {
    for (int i = 0; i < t.vecShape[0]; i++) {
      if (i > 0) os << endl;
      for (int j = 0; j < t.vecShape[1]; j++) {
        os << t.dosd[i][j] << " ";
      }
    }
  }
  return os;
}

int main () {
  Tensor t ({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  cout << t << endl << "---------------" << endl;
  Tensor t2 = Tensor::zeros ({3, 2});
  cout << t2 << endl << "---------------" << endl;
  Tensor t3 = Tensor::ones ({3, 3});
  cout << t3 << endl << "---------------" << endl;
  Tensor t4 = Tensor ::random ({10, 10}, 0.0, 1.0);
  cout << t4 << endl << "---------------" << endl;
  Tensor t5 = Tensor::arrange (1, 6);
  cout << t5 << endl << "---------------" << endl;
  Tensor t6 (t5);
  cout << t6 << endl << "---------------" << endl;
  Tensor t7 (move (t6));
  cout << t7 << endl << "---------------" << endl;
  cout << t6 << endl << "---------------" << endl;
  t6 = t7;
  cout << t6 << endl << "---------------" << endl;
}
