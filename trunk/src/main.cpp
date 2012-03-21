#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdio>
#include <bitset>

using namespace cv;
using namespace std;

#define forn(i,n) for(int (i)=0;i<int(n);++i)
#define forsn(i,s,n) for(int (i)=int(s);i<int(n);++i)

typedef complex<double> comp;

vector<comp> mivec;
vector<vector<comp> > miMat;
vector<vector<comp> > kerMatf(512,vector<comp>(512));
comp I(0.0, 1.0);
comp menosDos(-2.0,0.0);
comp PI(3.141592653589793,0.0);

vector<vector<bool> > calculado(530,vector<bool>(530,false));
vector<vector<comp> > valor(530, vector<comp>(530));

comp T(int N, int k) {
    if (calculado[N][k])
        return valor[N][k];
    calculado[N][k] = true;
    return valor[N][k] = exp(I * menosDos * PI * comp(k,0.0) / comp(N,0.0));
}

vector<int> lugar(512, -1);

int calc_lugar(int k) {
    if (lugar[k] >= 0)
        return lugar[k];
    int res = 0;
    forn (i,8) {
        res += k % 2;
        k >>= 1;
        res <<= 1;
    }

    return lugar[k] = res;
}

void fft(int baseT, int N) {
    if (N==1)
        return;
    int N2 = N / 2;
    int baseB = baseT+N2;
    fft(baseT,N2);
    fft(baseB,N2);
    forn (k,N2) {
        comp top = mivec[baseT+k];
        comp bot = mivec[baseB+k] * T(N, k);
        mivec[baseT+k] = top + bot;
        mivec[baseB+k] = top - bot;
    }
}

//void fft(int ini, int fin) {
//    vector<comp> res(fin,0);
//    //naive implementation
//    forn (i,fin) {
//        forn (j,fin) {
//            res[i] += mivec[j] * T(fin, (i * j) % fin);
//        }
//    }
//    forn (i,fin)
//        mivec[i] = res[i];
//}

void fft2() {
    mivec = vector<comp>(512);
    forn (x,512) {
        forn (y,512)
            mivec[lugar[y]] = miMat[y][x];
        fft(0, 512);
        forn (y,512)
            miMat[y][x] = mivec[y];
    }

    forn (y,512) {
        forn (x,512)
            mivec[lugar[x]] = miMat[y][x];
        fft(0,512);
        forn (x,512)
            miMat[y][x] = mivec[x];
    }
}

void ifft2() {
    I = -I;
    mivec = vector<comp>(512);
    forn (x,512) {
        forn (y,512)
            mivec[lugar[y]] = miMat[y][x];
        fft(0, 512);
        forn (y,512)
            miMat[y][x] = mivec[y] * (1.0 / 512.0);
    }

    forn (y,512) {
        forn (x,512)
            mivec[lugar[x]] = miMat[y][x];
        fft(0,512);
        forn (x,512)
            miMat[y][x] = mivec[x] * (1.0 / 512.0);
    }
    I = -I;
}

vector<vector<comp> > embossKernel() {
    vector<vector<comp> > res(512,vector<comp>(512,0));
    res[0][0] = -2;
    res[0][1] = -1;
    res[1][0] = -1;
    res[1][0] =  1;
    res[1][0] =  1;
    res[2][1] =  1;
    res[2][2] =  2;
    return res;
}

void copyMat(vector<vector<comp> >& mat1, Mat& mat2) {
    forn (y, 512)
        forn (x, 512)
            mat1[y][x] = comp(mat2.data[512*y+x],0.0);
}

void copyMat(vector<vector<comp> >& mat1, vector<vector<comp> >& mat2) {
    forn (y, 512)
        forn (x, 512)
            mat1[y][x] = mat2[y][x];
}

void multipCompMat(vector<vector<comp> >& mat1, vector<vector<comp> >& mat2) {
    forn (y, 512)
        forn (x, 512)
            mat1[y][x] *= mat2[y][x];
}

void embossFFT(Mat& m) {
    Mat res(m.size(),m.type());
    Size s = m.size();
    int w = s.width, h = s.height;

    miMat = embossKernel();

    fft2();

    copyMat(kerMatf, miMat);

    copyMat(miMat, m);
    fft2();
    multipCompMat(miMat,kerMatf);
    ifft2();

    forn (y, 512)
        forn (x, 512)
            m.data[512*(512-y)+(512-x)] = (unsigned char)(min(255,max(0,int(real(miMat[y][x])))));
}

Mat emboss(Mat& m) {
    Mat res(m.size(),m.type());
    Size s = m.size();
    int w = s.width, h = s.height;

    forsn (y,1,h-1) {
        forsn (x,1,w-1) {
            res.data[w*y+x] = min(255,max(0,
            - 2 * int(m.data[w*(y-1)+(x-1)]) - 1 * int(m.data[w*(y-1)+(x+0)])
            - 1 * int(m.data[w*(y+0)+(x-1)]) + 1 * int(m.data[w*(y+0)+(x+0)]) + 1 * int(m.data[w*(y+0)+(x+1)])
                                             + 1 * int(m.data[w*(y+1)+(x+0)]) + 2 * int(m.data[w*(y+1)+(x+1)])));
        }
    }
    return res;
}

int main()
{
    forn (i,512) {
        calc_lugar(i);
    }

    namedWindow("tp", 0);

    Mat imagen = imread("imgs/lena.bmp", 0);

    embossFFT(imagen);

    imshow("tp", imagen);

    waitKey();

    return 0;
}
