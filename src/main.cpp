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
const double pi = 3.141592653589793;

vector<vector<comp> > vec(2, vector<comp>(SIZE));
int usando = 1;
vector<vector<comp> > miMat(SIZE, vector<comp>(SIZE));
vector<vector<comp> > kernelF(SIZE,vector<comp>(SIZE)); // kernel in the frequency domain
comp I(0.0, 1.0);
comp menosDos(-2.0,0.0);
comp PI(pi,0.0);

vector<vector<vector<comp> > > T(2, vector<vector<comp> >(SIZE+100,vector<comp>(SIZE+100)));

void precalculateTwiddleFactors() {
    int N = SIZE;
    while (N > 1) {
        int N2 = N / 2;
        forn (k,N2) {
            T[0][N][k] = exp(I * menosDos * PI * comp(k,0.0) / comp(N,0.0));
            I = -I;
            T[1][N][k] = exp(I * menosDos * PI * comp(k,0.0) / comp(N,0.0));
            I = -I;
        }
        N = N2;
    }
}

inline comp Tf(int N, int k) {
    return exp(I * menosDos * PI * comp(k,0.0) / comp(N,0.0));
}

void fft(int base, int N, int inv) {
    if (N==1)
        return;
    int N2 = N / 2;
    forn (i,N2) {
        vec[1-usando][base+i] = vec[usando][base+2*i];
        vec[1-usando][base+N2+i] = vec[usando][base+2*i+1];
    }
    usando = 1 - usando;
    fft(base, N2, inv);
    fft(base+N2, N2, inv);

    forn (i,N2) { // "Butterfly"
        comp top = vec[1-usando][base+i];
        comp bot = vec[1-usando][base+N2+i] * T[inv][N][i];
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
        fft(0,SIZE,0);
        usando = 1-usando;
        forn (y,SIZE) {
            miMat[y][x] = vec[usando][y];
        }
    }

    forn (y,SIZE) {
        forn (x,SIZE) {
            vec[usando][x] = miMat[y][x];
        }
        fft(0,SIZE,0);
        usando = 1 - usando;
        forn (x,SIZE) {
            miMat[y][x] = vec[usando][x];
        }
    }
}

void ifft2() {
    const comp scale = comp(1.0 / float(SIZE), 0.0);
    forn (x,SIZE) {
        forn (y,SIZE)
            vec[usando][y] = miMat[y][x];
        fft(0,SIZE,1);
        usando = 1 - usando;
        forn (y,SIZE)
            miMat[y][x] = vec[usando][y] * scale;
    }

    forn (y,SIZE) {
        forn (x,SIZE)
            vec[usando][x] = miMat[y][x];
        fft(0,SIZE,1);
        usando = 1 - usando;
        forn (x,SIZE)
            miMat[y][x] = vec[usando][x] * scale;
    }
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
    multipCompMat(miMat, kernelF);
    ifft2();

    forn (y, SIZE) {
        forn (x, SIZE)
            m.data[SIZE*y+x] = (unsigned char)(min(255, max(0, int(real(miMat[y][x])))));
    }
}

int kernelSize = 5;

void recalculateGaussianKernel(int k, void* v) {
    vector<vector<float> > gauss(SIZE, vector<float>(SIZE,0.0));
    double scale = 1.0 / (2.0 * pi * kernelSize * kernelSize);
    double total = 0.0;
    forn(y,SIZE) {
        forn (x,SIZE) {
            gauss[y][x] = exp(-((x*x+y*y)/(2.0*kernelSize*kernelSize)));
            total += gauss[y][x];
        }
    }
    forn(y,SIZE) { // normalizamos para que el nivel de brillo no decaiga
        forn (x,SIZE) {
            gauss[y][x] /= total;
        }
    }

    fft2(miMat, gauss);
    copyMat(kernelF, miMat);
}

int main()
{
    precalculateTwiddleFactors();
    recalculateGaussianKernel(0,0);

    namedWindow("video", 0);
    createTrackbar("sigma", "video", &kernelSize, 200, &recalculateGaussianKernel);

    VideoCapture capCamera;
    Mat camFrame;
    Mat frame;
    Mat frame2(SIZE,SIZE,CV_8UC1);

    capCamera.open(0);

    for (;;) {
        capCamera >> camFrame; if(!camFrame.data) break;
        cvtColor(camFrame, frame, CV_RGB2GRAY);
        resize(frame, frame2, Size(SIZE,SIZE), 0, 0, INTER_LINEAR);
        embossFFT(frame2);
        imshow("video", frame2); if(waitKey(30) >= 0) break;
    }

    return 0;
}
