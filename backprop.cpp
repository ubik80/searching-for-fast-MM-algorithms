//Anpassung des Backpropagation Algorithmus für spezielle NN, wie in Master_Thesis_T_Spaeth.pdf beschrieben

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>


//für die Einbindung in Python
namespace py = pybind11;


//Backpropagation Algorithmus mit Ausmaskierung von Gewichten
//_Wa,_Wb,_Wc .. bisherige Gegwichtsmatrizen
//_Ma,_Mb,_Mc .. 0/1 Matrizen 0 bedeutet, Gewicht wird nicht verändert
//maxNumOfIters .. maximale Anzahl an Iterationen bevor Abbruch
//_nueAB,_nueC .. Lernraten
//tol .. Toleranz für Abbruch wegen Fehlerunterschreitung
auto backpropMasked(py::array_t<double> _Wa, py::array_t<double> _Wb,
                    py::array_t<double> _Wc, py::array_t<double> _Ma,
                    py::array_t<double> _Mb, py::array_t<double> _Mc,
                    int maxNumOfIters, double _nueAB, double _nueC,
                    double tol) {
  double nueAB = -_nueAB;
  double nueC = -_nueC;
  auto bufWa = _Wa.request();
  auto bufWb = _Wb.request();
  auto bufWc = _Wc.request();
  auto bufMa = _Ma.request();
  auto bufMb = _Mb.request();
  auto bufMc = _Mc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  double *Wa = (double *)bufWa.ptr;
  double *Wb = (double *)bufWb.ptr;
  double *Wc = (double *)bufWc.ptr;
  double *Ma = (double *)bufMa.ptr;
  double *Mb = (double *)bufMb.ptr;
  double *Mc = (double *)bufMc.ptr;
  std::vector<double> a(nn);
  std::vector<double> b(nn);
  std::vector<double> c(nn);
  std::vector<double> aWaveStar(p);
  std::vector<double> bWaveStar(p);
  std::vector<double> cWaveStar(p);
  std::vector<double> cWave(nn);
  std::vector<double> errC(nn);
  std::vector<double> errCStar(p);
  std::vector<double> deltaWa(nn * p);
  std::vector<double> deltaWb(nn * p);
  std::vector<double> deltaWc(nn * p);
  int errTolCnt = 0;

  auto seed =
      std::chrono::system_clock::now().time_since_epoch().count();  // seed
  std::default_random_engine dre(seed);                             // engine
  std::uniform_real_distribution<double> di(-1.0, 1.0);  // distribution

  for (auto iter = 0; iter < maxNumOfIters; iter++) {  // iteration
    std::generate(a.begin(), a.end(), [&] { return di(dre); });
    std::generate(b.begin(), b.end(), [&] { return di(dre); });
    double nA = sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0));
    double nB = sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));

    if (abs(nA) > 0.1 && abs(nB) > 0.1) {
      std::transform(a.begin(), a.end(), a.begin(),
                     std::bind1st(std::multiplies<double>(), 1.0 / nA));
      std::transform(b.begin(), b.end(), b.begin(),
                     std::bind1st(std::multiplies<double>(), 1.0 / nB));
      for (auto i = 0; i < n; i++) {  // C=A*B
        for (auto j = 0; j < n; j++) {
          c[i * n + j] = 0.0;
          for (auto k = 0; k < n; k++) {
            c[i * n + j] += a[i * n + k] * b[k * n + j];
          }                           // k
        }                             // j
      }                               // i
      for (auto i = 0; i < p; i++) {  // aWaveStar=Wa*a
        aWaveStar[i] = 0.0;
        bWaveStar[i] = 0.0;
        for (auto j = 0; j < nn; j++) {
          aWaveStar[i] += Wa[i * nn + j] * a[j];
          bWaveStar[i] += Wb[i * nn + j] * b[j];
        }  // j
        cWaveStar[i] = aWaveStar[i] * bWaveStar[i];
      }                                // i                         // i
      for (auto i = 0; i < nn; i++) {  // cWave=Wc*cWaveStar
        cWave[i] = 0.0;
        for (auto j = 0; j < p; j++) {
          cWave[i] += Wc[i * p + j] * cWaveStar[j];
        }  // j
      }    // i
      std::transform(cWave.begin(), cWave.end(), c.begin(), errC.begin(),
                     std::minus<double>());
      double errCNorm =
          sqrt(std::inner_product(errC.begin(), errC.end(), errC.begin(), 0.0));
      if (errCNorm > tol) {  // changed from 0.01
        errTolCnt = 0;
      } else {
        errTolCnt++;
      }
      for (auto i = 0; i < p; i++) {  // errCStar=Wc^T*errC
        errCStar[i] = 0.0;
        for (auto j = 0; j < nn; j++) {
          errCStar[i] += Wc[i + j * p] * errC[j];
        }                             // j
      }                               // i
      for (auto i = 0; i < p; i++) {  // dWa=-nue*outer(errCStar*bWaveStar,a)
        for (auto j = 0; j < nn; j++) {
          deltaWa[i * nn + j] = nueAB * errCStar[i] * bWaveStar[i] * a[j];
          deltaWb[i * nn + j] = nueAB * errCStar[i] * aWaveStar[i] * b[j];
          deltaWa[i * nn + j] *= Ma[i * nn + j];
          deltaWb[i * nn + j] *= Mb[i * nn + j];
          Wa[i * nn + j] += deltaWa[i * nn + j];  // update
          Wb[i * nn + j] += deltaWb[i * nn + j];  // update
        }                                         // j
      }                                           // i
      for (auto i = 0; i < nn; i++) {  // deltaWc=-nue*outer(errC,cWaveStar)
        for (auto j = 0; j < p; j++) {
          deltaWc[i * p + j] = nueC * errC[i] * cWaveStar[j];
          deltaWc[i * p + j] *= Mc[i * p + j];
          Wc[i * p + j] += deltaWc[i * p + j];  // update
        }                                       // j
      }                                         // i
      if (errTolCnt > 500) {
        for (auto i = 0; i < nn * p; i++) {
          if (isnan(Wa[i])) {
            return -1;
          }  // if (isnan(Wa[i]))
        }    // i
        return iter;
      }  // if
    }    // if (nA > 0.01 && nB > 0.01)
  }      // iter
  return -2;
}  // backpropMasked


//Backpropagation Algorithmus ohne Ausmaskierung von Gewichten
//_Wa,_Wb,_Wc .. bisherige Gegwichtsmatrizen
//maxNumOfIters .. maximale Anzahl an Iterationen bevor Abbruch
//_nueAB,_nueC .. Lernraten
//tol .. Toleranz für Abbruch wegen Fehlerunterschreitung
auto backpropNotMasked(py::array_t<double> _Wa, py::array_t<double> _Wb,
                       py::array_t<double> _Wc, int maxNumOfIters, double tol,
                       double _nueAB, double _nueC, int id) {
  double nueAB = -_nueAB;
  double nueC = -_nueC;
  auto bufWa = _Wa.request();
  auto bufWb = _Wb.request();
  auto bufWc = _Wc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  double *Wa = (double *)bufWa.ptr;
  double *Wb = (double *)bufWb.ptr;
  double *Wc = (double *)bufWc.ptr;
  std::vector<double> a(nn);
  std::vector<double> b(nn);
  std::vector<double> c(nn);
  std::vector<double> aWaveStar(p);
  std::vector<double> bWaveStar(p);
  std::vector<double> cWaveStar(p);
  std::vector<double> cWave(nn);
  std::vector<double> errC(nn);
  std::vector<double> errCStar(p);
  std::vector<double> deltaWa(nn * p);
  std::vector<double> deltaWb(nn * p);
  std::vector<double> deltaWc(nn * p);
  int errTolCnt = 0;

  auto seed =
      std::chrono::system_clock::now().time_since_epoch().count();  // seed
  std::default_random_engine dre(id + seed);                        // engine
  std::uniform_real_distribution<double> di(-1.0, 1.0);  // distribution

  for (auto iter = 0; iter < maxNumOfIters; iter++) {  // iteration
    std::generate(a.begin(), a.end(), [&] { return di(dre); });
    std::generate(b.begin(), b.end(), [&] { return di(dre); });
    double nA = sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0));
    double nB = sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));

    if (abs(nA) > 0.1 && abs(nB) > 0.1) {
      std::transform(a.begin(), a.end(), a.begin(),
                     std::bind1st(std::multiplies<double>(), 1.0 / nA));
      std::transform(b.begin(), b.end(), b.begin(),
                     std::bind1st(std::multiplies<double>(), 1.0 / nB));
      for (auto i = 0; i < n; i++) {  // C=A*B
        for (auto j = 0; j < n; j++) {
          c[i * n + j] = 0.0;
          for (auto k = 0; k < n; k++) {
            c[i * n + j] += a[i * n + k] * b[k * n + j];
          }                           // k
        }                             // j
      }                               // i
      for (auto i = 0; i < p; i++) {  // aWaveStar=Wa*a
        aWaveStar[i] = 0.0;
        bWaveStar[i] = 0.0;
        for (auto j = 0; j < nn; j++) {
          aWaveStar[i] += Wa[i * nn + j] * a[j];
          bWaveStar[i] += Wb[i * nn + j] * b[j];
        }  // j
        cWaveStar[i] = aWaveStar[i] * bWaveStar[i];
      }                                // i                         // i
      for (auto i = 0; i < nn; i++) {  // cWave=Wc*cWaveStar
        cWave[i] = 0.0;
        for (auto j = 0; j < p; j++) {
          cWave[i] += Wc[i * p + j] * cWaveStar[j];
        }  // j
      }    // i
      std::transform(cWave.begin(), cWave.end(), c.begin(), errC.begin(),
                     std::minus<double>());
      double errCNorm =
          sqrt(std::inner_product(errC.begin(), errC.end(), errC.begin(), 0.0));
      if (errCNorm > tol) {  // changed from 0.01
        errTolCnt = 0;
      } else {
        errTolCnt++;
      }
      for (auto i = 0; i < p; i++) {  // errCStar=Wc^T*errC
        errCStar[i] = 0.0;

        for (auto j = 0; j < nn; j++) {
          errCStar[i] += Wc[i + j * p] * errC[j];
        }                             // j
      }                               // i
      for (auto i = 0; i < p; i++) {  // dWa=-nue*outer(errCStar*bWaveStar,a)
        for (auto j = 0; j < nn; j++) {
          deltaWa[i * nn + j] = nueAB * errCStar[i] * bWaveStar[i] * a[j];
          deltaWb[i * nn + j] = nueAB * errCStar[i] * aWaveStar[i] * b[j];
          Wa[i * nn + j] += deltaWa[i * nn + j];  // update
          Wb[i * nn + j] += deltaWb[i * nn + j];  // update
        }                                         // j
      }                                           // i
      for (auto i = 0; i < nn; i++) {  // deltaWc=-nue*outer(errC,cWaveStar)
        for (auto j = 0; j < p; j++) {
          deltaWc[i * p + j] = nueC * errC[i] * cWaveStar[j];
          Wc[i * p + j] += deltaWc[i * p + j];  // update
        }                                       // j
      }                                         // i
      if (errTolCnt > 500) {                    
        for (auto i = 0; i < nn * p; i++) {
          if (isnan(Wa[i])) {
            return -1;
          }  // if (isnan(Wa[i]))
        }    // i
        return iter;
      }  // if
    }    // if (nA > 0.01 && nB > 0.01)
  }      // iter
  return -2;
}  // backpropNotMasked


//Bereitstellung in Python
PYBIND11_MODULE(backprop, m) {
  m.def("backpropMasked", backpropMasked);
  m.def("backpropNotMasked", backpropNotMasked);
}
