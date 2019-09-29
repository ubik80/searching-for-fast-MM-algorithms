#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace py = pybind11;

auto backpropM(py::array_t<double> _Wa, py::array_t<double> _Wb,
               py::array_t<double> _Wc, py::array_t<double> _Ma,
               py::array_t<double> _Mb, py::array_t<double> _Mc,
               int maxNumOfIters, double _nue, double tol) {
  double nue = -_nue;
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
      std::endl;
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
          deltaWa[i * nn + j] = nue * errCStar[i] * bWaveStar[i] * a[j];
          deltaWb[i * nn + j] = nue * errCStar[i] * aWaveStar[i] * b[j];
          deltaWa[i * nn + j] *= Ma[i * nn + j];
          deltaWb[i * nn + j] *= Mb[i * nn + j];
          Wa[i * nn + j] += deltaWa[i * nn + j];  // update
          Wb[i * nn + j] += deltaWb[i * nn + j];  // update
        }                                         // j
      }                                           // i
      for (auto i = 0; i < nn; i++) {  // deltaWc=-nue*outer(errC,cWaveStar)
        for (auto j = 0; j < p; j++) {
          deltaWc[i * p + j] = nue * errC[i] * cWaveStar[j];
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
}  // backprop

auto backprop(py::array_t<double> _Wa, py::array_t<double> _Wb,
              py::array_t<double> _Wc, int maxNumOfIters, double _nue,
              double tol) {
  double nue = -_nue;
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
          deltaWa[i * nn + j] = nue * errCStar[i] * bWaveStar[i] * a[j];
          deltaWb[i * nn + j] = nue * errCStar[i] * aWaveStar[i] * b[j];
          Wa[i * nn + j] += deltaWa[i * nn + j];  // update
          Wb[i * nn + j] += deltaWb[i * nn + j];  // update
        }                                         // j
      }                                           // i
      for (auto i = 0; i < nn; i++) {  // deltaWc=-nue*outer(errC,cWaveStar)
        for (auto j = 0; j < p; j++) {
          deltaWc[i * p + j] = nue * errC[i] * cWaveStar[j];
          Wc[i * p + j] += deltaWc[i * p + j];  // update
        }                                       // j
      }                                         // i
      if (errTolCnt > 500) {                    // changed from 500
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
}  // backprop

auto backpropNue(py::array_t<double> _Wa, py::array_t<double> _Wb,
                 py::array_t<double> _Wc, int maxNumOfIters, double tol,
                 double _nueAB, double _nueC) {
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
      if (errCNorm > tol) {
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
      if (errTolCnt > 500) {                    // changed from 500
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
}  // backprop

auto backpropNueM2(py::array_t<double> _Wa, py::array_t<double> _Wb,
                   py::array_t<double> _Wc, py::array_t<double> _Ma,
                   py::array_t<double> _Mb, py::array_t<double> _Mc,
                   int maxNumOfIters, double tol, double _nueA, double _nueB,
                   double _nueC, double jumpFac, int windowSize) {
  double nueA = -_nueA;
  double nueB = -_nueB;
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

  std::vector<double> randWa(nn * p);
  std::vector<double> randWb(nn * p);
  std::vector<double> randWc(nn * p);
  std::vector<double> errCHist(windowSize);
  int errCHistIdx = 0;
  int inBand = 0;
  double maxErrC;
  double minErrC;

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
          deltaWa[i * nn + j] = nueA * errCStar[i] * bWaveStar[i] * a[j];
          deltaWb[i * nn + j] = nueB * errCStar[i] * aWaveStar[i] * b[j];
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
      if (errTolCnt > 500) {                    // changed from 500
        for (auto i = 0; i < nn * p; i++) {
          if (isnan(Wa[i]) || isnan(Wb[i] || isnan(Wc[i]))) {
            return -1;
          }  // if (isnan(Wa[i]))
        }    // i
        return iter;
      }  // if

      maxErrC = *std::max_element(errCHist.begin(), errCHist.end());
      minErrC = *std::min_element(errCHist.begin(), errCHist.end());
      errCHist[errCHistIdx] = errCNorm;
      errCHistIdx = (errCHistIdx + 1) % windowSize;
      if (errCNorm < maxErrC && errCNorm > minErrC) {
        inBand += 1;
      } else {
        inBand = 0;
      }
      if (inBand > windowSize) {
        std::generate(randWa.begin(), randWa.end(), [&] { return di(dre); });
        std::generate(randWb.begin(), randWb.end(), [&] { return di(dre); });
        std::generate(randWc.begin(), randWc.end(), [&] { return di(dre); });
        for (auto i = 0; i < p; i++) {
          for (auto j = 0; j < nn; j++) {
            Wa[i * nn + j] += randWa[i * nn + j] * jumpFac;
            Wb[i * nn + j] += randWb[i * nn + j] * jumpFac;
            Wc[j * p + i] += randWc[j * p + i] * jumpFac;
          }  // for j
        }    // for i
      }      // if (inBand > windowSize)
    }        // if (nA > 0.01 && nB > 0.01)
  }          // iter

  return -2;
}  // backprop

auto backpropNueRND(py::array_t<double> _Wa, py::array_t<double> _Wb,
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
      if (errTolCnt > 500) {                    // changed from 500
        for (auto i = 0; i < nn * p; i++) {
          if (isnan(Wa[i])) {
            // std::cout << "... NAN at iter: " << (int)iter << std::endl;
            return -1;
          }  // if (isnan(Wa[i]))
        }    // i
        // std::cout << "... finished, iters: " << (int)iter << std::endl;
        return iter;
      }  // if
    }    // if (nA > 0.01 && nB > 0.01)
  }      // iter
  return -2;
}  // backprop

PYBIND11_MODULE(backprop, m) {
  m.def("backpropM", backpropM);
  m.def("backprop", backprop);
  m.def("backpropNue", backpropNue);
  m.def("backpropNueM2", backpropNueM2);
  m.def("backpropNueRND", backpropNueRND);
}
