#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <complex>
#include <vector>

using namespace cv;
using namespace std;

#define forn(i,n) for(int (i)=0;i<int(n);++i)

typedef complex<float> comp;

const int SIZE = 512;
const double pi = 3.141592653589793;

int kernelSize = 50;
int usando = 1;

vector<vector<comp> > vec(2, vector<comp>(SIZE));
vector<vector<comp> > miMat(SIZE, vector<comp>(SIZE));
vector<vector<comp> > kernelF(SIZE,vector<comp>(SIZE)); // kernel in the frequency domain
vector<vector<vector<comp> > > T(2, vector<vector<comp> >(SIZE+10,vector<comp>(SIZE+10)));

void precalculateTwiddleFactors() {
    comp I(0.0, 1.0), menosDos(-2.0,0.0), PI(pi,0.0);
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

void fft2(vector<vector<comp> >& miMat, vector<vector<comp> >& m) {
    forn (x,SIZE) {
        forn (y,SIZE) {
            vec[usando][y] = m[y][x];
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

void copyMatInterc(vector<vector<comp> >& mat1, Mat& mat2) {
    int N2 = SIZE / 2;
    forn (y, N2)
        forn (x, SIZE)
            mat1[y][x] = comp(mat2.data[SIZE*y+x],mat2.data[SIZE*(y+N2)+x]);
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

void conv2d(Mat& m) {
    vector<vector<comp> > mf(SIZE,vector<comp>(SIZE));

    copyMat(mf, m);

    fft2(miMat, mf);
    multipCompMat(miMat, kernelF);
    ifft2();

    forn (y, SIZE) {
        forn (x, SIZE) {
            m.data[SIZE*y+x] = (unsigned char)(min(255, max(0, int(real(miMat[y][x])))));
        }
    }
}

void recalculateGaussianKernel(int k, void* v) {
    kernelSize = max(1,kernelSize);
    vector<vector<comp> > gauss(SIZE, vector<comp>(SIZE,0.0));
    double total = 0.0;
    double maxim = 0.0;
    forn(y,SIZE) { // TODO: aprovechar separabilidad y simetría de la gaussiana
        forn (x,SIZE) {
            int varx = (x > SIZE / 2) ? SIZE-x : x;
            int vary = (y > SIZE / 2) ? SIZE-y : y;
            double sumando = exp(-(varx*varx+vary*vary)/(2.0*kernelSize*kernelSize));
            maxim = max(maxim, sumando);
            gauss[y][x] = sumando;
            total += sumando;
        }
    }
    forn(y,SIZE) { // normalizamos para que el nivel de brillo no decaiga
        forn (x,SIZE) {
            gauss[y][x].real() /= total;
        }
    }

    double escala = maxim / total;
    Mat muestraKernel(SIZE, SIZE, CV_8UC1);
    forn (y,SIZE)
        forn (x, SIZE)
            muestraKernel.data[y*SIZE+x] = (unsigned char)(255.0 * real(gauss[y][x]) / escala);

    imshow("kernel", muestraKernel);

    fft2(miMat, gauss);
    copyMat(kernelF, miMat);
}

int main()
{
    precalculateTwiddleFactors();
    namedWindow("kernel", 0);
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
        conv2d(frame2);
        imshow("video", frame2); if(waitKey(30) >= 0) break;
    }

    return 0;
}
