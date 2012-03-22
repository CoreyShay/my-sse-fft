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

vector<vector<comp> > vec(2, vector<comp>(SIZE));
int usando = 1;
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

void fft(int base, int N) {
    if (N==1)
        return;
    int N2 = N / 2;
    forn (i,N2) {
        vec[1-usando][base+i] = vec[usando][base+2*i];
        vec[1-usando][base+N2+i] = vec[usando][base+2*i+1];
    }
    usando = 1 - usando;
    fft(base, N2);
    fft(base+N2, N2);

    forn (i,N2) { // "Butterfly"
        comp top = vec[1-usando][base+i];
        comp bot = vec[1-usando][base+N2+i] * T(N, i);
        vec[usando][base+i] = top + bot;
        vec[usando][base+N2+i] = top - bot;
    }
    usando = 1 - usando;
}

void fft2(vector<vector<comp> >& miMat, vector<vector<float> >& m) {
    forn (x,SIZE) {
        forn (y,SIZE) {
            vec[usando][y] = comp(m[y][x], 0.0);
        }
        fft(0,SIZE);
        usando = 1-usando;
        forn (y,SIZE) {
            miMat[y][x] = vec[usando][y];
        }
    }

    forn (y,SIZE) {
        forn (x,SIZE) {
            vec[usando][x] = miMat[y][x];
        }
        fft(0,SIZE);
        usando = 1 - usando;
        forn (x,SIZE) {
            miMat[y][x] = vec[usando][x];
        }
    }
}

void ifft2() {
    I = -I;
    const comp scale = comp(1.0 / float(SIZE), 0.0);
    forn (x,SIZE) {
        forn (y,SIZE)
            vec[usando][y] = miMat[y][x];
        fft(0,SIZE);
        usando = 1 - usando;
        forn (y,SIZE)
            miMat[y][x] = vec[usando][y] * scale;
    }

    forn (y,SIZE) {
        forn (x,SIZE)
            vec[usando][x] = miMat[y][x];
        fft(0,SIZE);
        usando = 1- usando;
        forn (x,SIZE)
            miMat[y][x] = vec[usando][x] * scale;
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
    namedWindow("video", 0);

    //Mat im1 = imread("imgs/lena.bmp", 0);
    VideoCapture capCamera;
    Mat camFrame;
    Mat frame;
    Mat frame2(SIZE,SIZE,CV_8UC1);

    vector<vector<float> > embker = embossKernel();
    fft2(miMat, embker);
    copyMat(kerMatf, miMat);

    capCamera.open(0);
    capCamera >> camFrame;

    for (;;) {
        capCamera >> camFrame; if(!camFrame.data) break;
        cvtColor(camFrame, frame, CV_RGB2GRAY);
        resize(frame, frame2, Size(SIZE,SIZE), 0, 0, INTER_LINEAR);
        embossFFT(frame2);
        imshow("video", frame2); if(waitKey(30) >= 0) break;
    }

    return 0;
}
