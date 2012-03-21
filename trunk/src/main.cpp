#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <complex>
#include <vector>

using namespace cv;
using namespace std;

#define forn(i,n) for(int (i)=0;i<int(n);++i)
#define forsn(i,s,n) for(int (i)=int(s);i<int(n);++i)

typedef complex<float> comp;

const int SIZE = 512;

vector<comp> mivec(SIZE);
vector<vector<comp> > miMat(SIZE, vector<comp>(SIZE));
vector<vector<comp> > kerMatf(SIZE,vector<comp>(SIZE));
comp I(0.0, 1.0);
comp menosDos(-2.0,0.0);
comp PI(3.141592653589793,0.0);

vector<vector<bool> > calculado(SIZE+30,vector<bool>(SIZE+30,false));
vector<vector<comp> > valor(SIZE+30, vector<comp>(SIZE+30));

inline comp T(int N, int k) {
    if (calculado[N][k])
        return valor[N][k];
    calculado[N][k] = true;
    return valor[N][k] = exp(I * menosDos * PI * comp(k,0.0) / comp(N,0.0));
}

void fft(vector<comp>& vec) {
    int N = vec.size();
    if (N==1)
        return;
    int N2 = N / 2;
    vector<comp> vEven(N2), vOdd(N2);
    forn (i,N2) {
        vEven[i] = vec[2*i];
        vOdd[i] = vec[2*i+1];
    }
    fft(vEven);
    fft(vOdd);

    forn (k,N2) { // "Butterfly"
        comp top = vEven[k];
        comp bot = vOdd[k] * T(N, k);
        vec[k] = top + bot;
        vec[N2+k] = top - bot;
    }
}

void fft2(vector<vector<comp> >& miMat, vector<vector<float> >& m) {
//    forn (x,SIZE/2) {
//        forn (y,SIZE) { // Process two columns with one FFT
//            mivec[y] = comp(m.data[y*SIZE+2*x], m.data[y*SIZE+2*x+1]);
//        }
//        fft(mivec);
//        forn (y,SIZE) {
//            miMat[y][2*x] = real(mivec[y]);
//            miMat[y][2*x+1] = imag(mivec[y]);
//        }
//    }

    forn (x,SIZE) {
        forn (y,SIZE) {
            mivec[y] = comp(m[y][x], 0.0);
        }
        fft(mivec);
        forn (y,SIZE) {
            miMat[y][x] = mivec[y];
        }
    }

    forn (y,SIZE) {
        forn (x,SIZE) {
            mivec[x] = miMat[y][x];
        }
        fft(mivec);
        forn (x,SIZE) {
            miMat[y][x] = mivec[x];
        }
    }
}

void ifft2() {
    I = -I;
    mivec = vector<comp>(SIZE);
    const comp scale = comp(1.0 / float(SIZE), 0.0);
    forn (x,SIZE) {
        forn (y,SIZE)
            mivec[y] = miMat[y][x];
        fft(mivec);
        forn (y,SIZE)
            miMat[y][x] = mivec[y] * scale;
    }

    forn (y,SIZE) {
        forn (x,SIZE)
            mivec[x] = miMat[y][x];
        fft(mivec);
        forn (x,SIZE)
            miMat[y][x] = mivec[x] * scale;
    }
    I = -I;
}

vector<vector<float> > embossKernel() {
    vector<vector<float> > res(SIZE,vector<float>(SIZE,0.0));
    res[0][0] = -2.0;
    res[0][1] = -1.0;
    res[1][0] = -1.0;
    res[1][1] = 1.0;
    res[1][2] = 1.0;
    res[2][1] = 1.0;
    res[2][2] = 2.0;
    return res;
}

void copyMat(vector<vector<float> >& mat1, Mat& mat2) {
    forn (y, SIZE)
        forn (x, SIZE)
            mat1[y][x] = mat2.data[SIZE*y+x];
}

void copyMat(vector<vector<comp> >& mat1, Mat& mat2) {
    forn (y, SIZE)
        forn (x, SIZE)
            mat1[y][x] = comp(mat2.data[SIZE*y+x],0.0);
}

void copyMat(vector<vector<comp> >& mat1, vector<vector<comp> >& mat2) {
    forn (y, SIZE)
        forn (x, SIZE)
            mat1[y][x] = mat2[y][x];
}

void multipCompMat(vector<vector<comp> >& mat1, vector<vector<comp> >& mat2) {
    forn (y, SIZE)
        forn (x, SIZE)
            mat1[y][x] *= mat2[y][x];
}

void embossFFT(Mat& m) {
    Size s = m.size();
    int w = s.width, h = s.height;

    vector<vector<float> > embker = embossKernel();

    fft2(miMat, embker);

    copyMat(kerMatf, miMat);

    vector<vector<float> > mf(SIZE,vector<float>(SIZE));

    copyMat(mf, m);

    fft2(miMat, mf);
    multipCompMat(miMat, kerMatf);
    ifft2();

    forn (y, SIZE)
        forn (x, SIZE)
            m.data[SIZE*(SIZE-y)+(SIZE-x)] = (unsigned char)(min(255, max(0, int(real(miMat[y][x])))));
}

int main()
{
    namedWindow("tp", 0);

    Mat im1 = imread("imgs/lena.bmp", 0);

    Mat imagen = im1(Range(0,SIZE), Range(0,SIZE));

    embossFFT(imagen);

    imshow("tp", imagen);

    waitKey();

    return 0;
}
