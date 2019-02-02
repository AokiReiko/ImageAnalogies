//
//  main.cpp
//  ImageAnalogies
//
//  Created by aokireiko on 18/12/1.
//  Copyright © 2018年 aokireiko. All rights reserved.
//

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.hpp>
#include "opencv2/core/core.hpp"
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <fstream>
#include <stdio.h>

#include "ann_match.hpp"
#include "coherence_match.hpp"

#define LEVEL 3
#define FINE_WINDOW 5
#define COARSE_WINDOW 3
#define CWIN_SIZE COARSE_WINDOW*COARSE_WINDOW
#define FWIN_SIZE FINE_WINDOW*FINE_WINDOW

typedef flann::Matrix<float> fMatrix;
typedef flann::Matrix<int> iMatrix;
typedef flann::L2<float> fL2;
typedef flann::Index<flann::L2<float>> Index;

using namespace std;
using namespace cv;

/*
 1. Convert to YIQ form and Y as the feature.
 2. Luminence remapping. (TODO)
 4. opencv的坐标系是(y,x): Mat::at(Point(x, y)) == Mat::at(y,x)
 3. Mat类型数据也是按行存储的,坐标系上面相同的y坐标构成1行，+cols.
 4. 权重的处理，应该放在A的feature中，以及normalize(TODO)
 */

int f_len = 2*CWIN_SIZE + ceil(1.5*FWIN_SIZE);

void remapping(Mat &A, const Mat &B) {
    Scalar mean_b, mean_a, stddev_b, stddev_a;
    meanStdDev(A, mean_a, stddev_a);
    meanStdDev(B, mean_b, stddev_b);
    for (int j = 0; j < A.rows; j++) {
        for (int i = 0; i < A.cols; i++) {
            float a = A.at<float>(j,i);
            A.at<float>(j,i) = (a-mean_a[0])*stddev_b[0]/stddev_a[0] + mean_b[0];
        }
    }
    
}
//CWIN_SIZE + CWIN_SIZE + FWIN_SIZE + FWIN_SIZE/2+1
float weighted_features(const float* a, const float* b, const float * weights) {
    float res = 0;
    for (int i = 0; i < 2*CWIN_SIZE; i++) {
        res += (a[i] - b[i])*(a[i] - b[i]); // weights[i]*
    }
    
    for (int i = 2*CWIN_SIZE; i < f_len; i++) {
        res += (a[i] - b[i])*(a[i] - b[i]); // weights[i]*
    }
    return res;
}

void compute_feature(float* res, int j, int i , Mat &m, Mat &mp, Mat &coarse_m, Mat &coarse_mp, float* kernel) {
    int border_x = m.cols, border_y = m.rows, na = border_x * border_y;
    int pborder_x = coarse_m.cols, pborder_y = coarse_m.rows;
    for (int i = 0; i < f_len; i++) {
        res[i] = 0;
    }
    // For borders, we pad these "lost" neighborhoods with constant "0".
    // features: coarse windows of Ai-1 and A'i-1 + fine windows of Ai and A'i
    
    int res_ind = 0;
    int ii = i/2, jj = j/2;
    float sum = 0;
    for (int k = -COARSE_WINDOW/2; k <= COARSE_WINDOW/2; k++) {
        for (int t = -COARSE_WINDOW/2; t <= COARSE_WINDOW/2; t++) {
            int xx = ii+t, yy = jj+k;
            res_ind ++;
            if (xx < 0 || yy < 0 || xx >= pborder_x || yy >= pborder_y) continue;
            res[res_ind-1] = coarse_m.at<float>(yy, xx) * kernel[res_ind-1];
            sum += res[res_ind-1] * res[res_ind-1];
            res[res_ind-1 + CWIN_SIZE] = coarse_mp.at<float>(yy, xx) * kernel[res_ind-1 + CWIN_SIZE];
            sum += res[res_ind-1 + CWIN_SIZE] * res[res_ind-1 + CWIN_SIZE];

        }
                
    }
    res_ind += CWIN_SIZE;
    int L_nil = 0, L_max = ceil(0.5*FWIN_SIZE);
    for (int k = -FINE_WINDOW/2; k <= FINE_WINDOW/2; k++) {
        for (int t = -FINE_WINDOW/2; t <= FINE_WINDOW/2; t++) {
            int xx = i+t, yy = j+k;
            res_ind ++;
            L_nil++;
            if (xx < 0 || yy < 0 || xx >= border_x || yy >= border_y) continue;
            res[res_ind-1] = m.at<float>(yy, xx) * kernel[res_ind-1];
            sum += res[res_ind-1] * res[res_ind-1];
            if (L_nil > L_max) continue;
            res[res_ind-1 + FWIN_SIZE] = mp.at<float>(yy, xx) * kernel[res_ind-1+FWIN_SIZE];
            sum += res[res_ind-1 + FWIN_SIZE] * res[res_ind-1 + FWIN_SIZE];
        }
    }
    res_ind = 0;
    if (sum == 0) return;
    for (int k = 0; k < f_len; k++) {
        res[res_ind++] /= sum;
    }
}
void compute_feature(float* res, int j, int i , Mat &m, Mat &mp, float* kernel) {
    int border_x = m.cols, border_y = m.rows;
    for (int i = 0; i < f_len; i++) {
        res[i] = 0;
    }
    // For borders, we pad these "lost" neighborhoods with constant "0".
    // features: coarse windows of Ai-1 and A'i-1 + fine windows of Ai and A'i
    
    int res_ind = 2*CWIN_SIZE;
   
    int L_nil = 0, L_max = ceil(0.5*FWIN_SIZE);
    
    float sum = 0;
    for (int k = -FINE_WINDOW/2; k <= FINE_WINDOW/2; k++) {
        for (int t = -FINE_WINDOW/2; t <= FINE_WINDOW/2; t++) {
            int xx = i+t, yy = j+k;
            res_ind ++;
            L_nil++;
            if (xx < 0 || yy < 0 || xx >= border_x || yy >= border_y) continue;
            res[res_ind-1] = m.at<float>(yy, xx) * kernel[res_ind-1];
            sum += res[res_ind-1] * res[res_ind-1];
            if (L_nil > L_max) continue;
            res[res_ind-1 + FWIN_SIZE] = mp.at<float>(yy, xx) * kernel[res_ind-1+FWIN_SIZE];
            sum += res[res_ind-1 + FWIN_SIZE] * res[res_ind-1 + FWIN_SIZE];
        }
    }
    
    res_ind = 0;
    if (sum == 0) return;
    for (int k = 0; k < f_len; k++) {
        res[res_ind++] /= sum;
    }
    
    
}

/* Features 按行（y坐标相同）存储*/
float* pad_A_features(Mat &m, Mat &mp, Mat &coarse_m, Mat &coarse_mp, float* kernel) {
    
    int border_x = m.cols, border_y = m.rows, na = border_x * border_y;
    int pborder_x = coarse_m.cols, pborder_y = coarse_m.rows;
    float* res = new float[na * f_len];
    memset(res, 0, na * f_len * sizeof(float));
    // For borders, we pad these "lost" neighborhoods with constant "0".
    // features: coarse windows of Ai-1 and A'i-1 + fine windows of Ai and A'i
    for (int j = 0; j < border_y; j++) {
        for (int i = 0; i < border_x; i++) {
            int ind = j * border_x + i;
            int res_ind = ind * f_len;
            int ii = i/2, jj = j/2;
            int k_i = 0;
            float sum = 0;
            for (int k = -COARSE_WINDOW/2; k <= COARSE_WINDOW/2; k++) {
                for (int t = -COARSE_WINDOW/2; t <= COARSE_WINDOW/2; t++) {
                    int xx = ii+t, yy = jj+k;
                    res_ind ++;
                    k_i++;
                    if (xx < 0 || yy < 0 || xx >= pborder_x || yy >= pborder_y) continue;
                    res[res_ind-1] = coarse_m.at<float>(yy, xx) * kernel[k_i-1];
                    sum += res[res_ind-1] * res[res_ind-1];
                    res[res_ind-1 + CWIN_SIZE] = coarse_mp.at<float>(yy, xx) * kernel[k_i-1 + CWIN_SIZE];
                    sum += res[res_ind-1 + CWIN_SIZE] * res[res_ind-1 + CWIN_SIZE];
                }
                
            }
            res_ind += CWIN_SIZE;
            k_i += CWIN_SIZE;
            int L_nil = 0, L_max = ceil(0.5*FWIN_SIZE);
            for (int k = -FINE_WINDOW/2; k <= FINE_WINDOW/2; k++) {
                for (int t = -FINE_WINDOW/2; t <= FINE_WINDOW/2; t++) {
                    int xx = i+t, yy = j+k;
                    res_ind ++;
                    k_i++;
                    L_nil++;
                    if (xx < 0 || yy < 0 || xx >= border_x || yy >= border_y) continue;
                    res[res_ind-1] = m.at<float>(yy, xx) * kernel[k_i-1];
                    sum += res[res_ind-1] * res[res_ind-1];
                    if (L_nil > L_max) continue;
                    res[res_ind-1 + FWIN_SIZE] = mp.at<float>(yy, xx) * kernel[k_i-1 + FWIN_SIZE];
                    sum += res[res_ind-1 + FWIN_SIZE] * res[res_ind-1 + FWIN_SIZE];
                }
                
            }
            res_ind = ind * f_len;
            if (sum == 0) continue;
            for (int k = 0; k < f_len; k++) {
                res[res_ind++] /= sum;
            }
        }
    }
    return res;
    
}
float* pad_A_features(Mat &m, Mat &mp, float* kernel) {
    int border_x = m.cols, border_y = m.rows, na = border_x * border_y;
    float* res = new float[na * f_len];
    memset(res, 0, na * f_len * sizeof(float));
    // For borders, we pad these "lost" neighborhoods with constant "0".
    // features: coarse windows of Ai-1 and A'i-1 + fine windows of Ai and A'i
    for (int j = 0; j < border_y; j++) {
        for (int i = 0; i < border_x; i++) {
            int ind = j * border_x + i;
            int res_ind = ind * f_len;
            int k_i = 0;
            res_ind += 2*CWIN_SIZE;
            k_i += 2*CWIN_SIZE;
            float sum = 0;
            int L_nil = 0, L_max = ceil(0.5*FWIN_SIZE);
            for (int k = -FINE_WINDOW/2; k <= FINE_WINDOW/2; k++) {
                for (int t = -FINE_WINDOW/2; t <= FINE_WINDOW/2; t++) {
                    int xx = i+t, yy = j+k;
                    res_ind ++;
                    k_i++;
                    L_nil++;
                    if (xx < 0 || yy < 0 || xx >= border_x || yy >= border_y) continue;
                    res[res_ind-1] = m.at<float>(yy, xx) * kernel[k_i-1];
                    sum += res[res_ind-1] * res[res_ind-1];
                    if (L_nil > L_max) continue;
                    res[res_ind-1 + FWIN_SIZE] = mp.at<float>(yy, xx) * kernel[k_i-1 + FWIN_SIZE];
                    sum += res[res_ind-1 + FWIN_SIZE] * res[res_ind-1 + FWIN_SIZE];
                    
                }
                
            }
            
            res_ind = ind * f_len;
            if (sum == 0) continue;
            for (int k = 0; k < f_len; k++) {
                res[res_ind++] /= sum;
            }
            
            
        }
    }
    return res;
    
}


void get_gaussian_kernel(float *gaus, const int size,const double sigma)
{
    const float PI=4.0*atan(1.0); //圆周率π赋值
    int center=size/2;
    float sum=0;
    int ind = 0;
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[ind]=(1/(2*PI*sigma*sigma))*exp(-((0.0+i-center)*(0.0+i-center)+(j-center)*(j-center))/(2*sigma*sigma));
            sum+=gaus[ind];
            ind++;
        }
    }
    ind = 0;
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[ind++]/=sum;
        }
    }
    return ;
}

int main( void ){
    cout << "Color it? (y or n): ";
    bool tocolor = false;
    char c;
    cin >> c;
    if (c == 'y') tocolor = true;
    double ka;
    cout << "Input ka (2-25 for non-photorealistic filters, 1 for line filters, 0.5-5 for texture):";
    cin >> ka;
    
    
   /*
    float test[24] = {0,-1,5,6, 10,2,1,1,1,1,1,1,1,2,3,4, 0,0,0,0,0,0,0,-1};
    fMatrix mm(test, 12,2);
    
    cout << *mm.ptr() << " " << *(mm.ptr()+3) << " " <<*(mm.ptr()+2) << endl;
    AnnMatch aa(mm, 12,2);
    
    cout << "built"<<endl;
    int rre[2];
    float ddi[2];
    iMatrix re(new int[2], 1,3);
    fMatrix di(new float[2],1,3);
    float qq[2] = {10,0.5};
    for (int k = 0; k < 100000; k++) {
        float qu[2];
        qu[0] = k;
        qu[1] = k+1;
        aa.best_ann_match(re, di, qu, 2);
        //for (int i = 0; i < 1; i++)
          //  cout << k << " complete " <<(*re.ptr()+i) << " " << (*di.ptr()+i)<<  " "<<endl;
    }
    delete [] re.ptr();
    delete [] di.ptr();
    
    int size=5; //定义卷积核大小
    double **gaus=new double *[size];
    for(int i=0;i<size;i++)
    {
        gaus[i]=new double[size];  //动态生成矩阵
    }
    cout<<"尺寸 = 3*3，Sigma = 1，高斯卷积核参数为："<<endl;

    return 0;
    */
    
    
    Mat A_pyramid[LEVEL], AP_pyramid[LEVEL], B_pyramid[LEVEL], BP_pyramid[LEVEL], AP_src[LEVEL];
    Mat img, yiq[3];
    fMatrix *A_f[LEVEL];
    string a_path, ap_path, b_path;
    cout << " Input A path(./images/+): ";
    cin >> a_path;
    img = imread("./images/"+a_path, CV_LOAD_IMAGE_COLOR);
    if (! img.data) {
        cout << "No such image" << endl;
        return 0;
    }
    cvtColor(img, img, CV_BGR2YUV);
    split(img, yiq);
    yiq[0].convertTo(A_pyramid[LEVEL-1], CV_32FC1);
    
    cout << " Input A' path(./images/+): ";
    cin >> ap_path;
    img = imread("./images/"+ap_path, CV_LOAD_IMAGE_COLOR);
    AP_src[LEVEL-1] = imread("./images/"+ap_path, CV_LOAD_IMAGE_COLOR);
    
    if (! img.data) {
        cout << "No such image" << endl;
        return 0;
    }
    cvtColor(img, img, CV_BGR2YUV);
    split(img, yiq);
    yiq[0].convertTo(AP_pyramid[LEVEL-1], CV_32FC1);
    
    
    cout << " Input B path(./images/+): ";
    cin >> b_path;
    img = imread("./images/"+b_path, CV_LOAD_IMAGE_COLOR);
    if (! img.data) {
        cout << "No such image" << endl;
        return 0;
    }
    cvtColor(img, img, CV_BGR2YUV);
    split(img, yiq);
    yiq[0].convertTo(B_pyramid[LEVEL-1], CV_32FC1);
    B_pyramid[LEVEL-1].copyTo(BP_pyramid[LEVEL-1]);
    /*
    
    cout << CV_32FC1 << " " << CV_8UC1 << endl;
    cout << FWIN_SIZE<<endl;
    cout << A_pyramid[LEVEL-1].type() << endl;
    cout << *(A_pyramid[LEVEL-1].ptr<float>(0, 1)) << " " << A_pyramid[LEVEL-1].at<float>(0,1) << endl;
    cout << *(A_pyramid[LEVEL-1].ptr<float>(1, 0)+10) << " " << A_pyramid[LEVEL-1].at<float>(1,1) << endl;
    cout << *(A_pyramid[LEVEL-1].ptr<float>(1, 10)) << " " << A_pyramid[LEVEL-1].at<float>(1,1) << endl;
    cout << *(A_pyramid[LEVEL-1].ptr<float>(0, 1)+1) << " " << A_pyramid[LEVEL-1].at<float>(0,2) << endl;
    */
    
    // Remapping A and A'
    //remapping(A_pyramid[LEVEL-1], B_pyramid[LEVEL-1]);
    //remapping(AP_pyramid[LEVEL-1], B_pyramid[LEVEL-1]);
    float *gauss_weight = new float[2*(CWIN_SIZE+FWIN_SIZE)];
    double sigma = 0.8;
    get_gaussian_kernel(gauss_weight, COARSE_WINDOW, sigma);
    get_gaussian_kernel(gauss_weight+CWIN_SIZE, COARSE_WINDOW, sigma);
    get_gaussian_kernel(gauss_weight+2*CWIN_SIZE, FINE_WINDOW, sigma);
    get_gaussian_kernel(gauss_weight+2*CWIN_SIZE+FWIN_SIZE, FINE_WINDOW, sigma);
    
    
    // Build pyramids.
    for (int i = LEVEL-2; i >= 0; i--) {
        pyrDown(A_pyramid[i+1], A_pyramid[i], Size( A_pyramid[i+1].cols/2, A_pyramid[i+1].rows/2 ));
        pyrDown(AP_pyramid[i+1], AP_pyramid[i], Size( AP_pyramid[i+1].cols/2, AP_pyramid[i+1].rows/2 ));
        pyrDown(B_pyramid[i+1], B_pyramid[i], Size( B_pyramid[i+1].cols/2, B_pyramid[i+1].rows/2 ));
        B_pyramid[i].copyTo(BP_pyramid[i]);
        pyrDown(AP_src[i+1], AP_src[i], Size( A_pyramid[i+1].cols/2, A_pyramid[i+1].rows/2 ));
    }
    cout << "Build pyramids completed." << endl;

     // Compute features for (A,A') and (B,B'). The dimension is 3*3*2+5*5+13(L)=56
    cout << "Feature's length is " << f_len << endl;
    for (int i = 0; i < LEVEL; i++) {
        float* features = NULL;
        int na = A_pyramid[i].cols * A_pyramid[i].rows;
        if (i == 0) features = pad_A_features(A_pyramid[i], AP_pyramid[i], gauss_weight);
        else features = pad_A_features(A_pyramid[i], AP_pyramid[i], A_pyramid[i-1], AP_pyramid[i-1], gauss_weight);
        A_f[i] = new fMatrix(features, na, f_len);
    }
    cout << "A's features are computed." << endl;
    
    // From coarsest to finest
    for (int l = 0; l < LEVEL; l++) {
        float w = 1 + pow(2, l-LEVEL+1) * ka;
        Mat Bl(B_pyramid[l].rows, B_pyramid[l].cols, CV_8UC3);
        
        
        Index<fL2> ann(*A_f[l], KDTreeIndexParams(4));
        ann.buildIndex();
        CoherenceMatch cm(A_pyramid[l].cols, A_pyramid[l].rows, B_pyramid[l].cols, B_pyramid[l].rows);
        
        
        iMatrix indices(new int[f_len], 1, f_len);
        fMatrix dists(new float[f_len], 1, f_len);
        float *query = new float[f_len];
        int border_x = B_pyramid[l].cols, border_y = B_pyramid[l].rows;
        int na = border_x*border_y;
        int *s = new int[na];
        memset(s, 0, na*sizeof(int));
        
        for (int j = 0; j < border_y; j++) {
            for (int i = 0; i < border_x; i++) {
                if (l == 0) compute_feature(query, j, i, B_pyramid[l], BP_pyramid[l], gauss_weight);
                else compute_feature(query, j, i, B_pyramid[l], BP_pyramid[l], B_pyramid[l-1], BP_pyramid[l-1], gauss_weight);
                
                fMatrix q(query, 1, f_len);
                ann.knnSearch(q, indices, dists, 1, SearchParams(128));
                //am.best_ann_match(indices, dists, query, f_len);
                int p_app = *indices.ptr();
                int p_coh = cm.best_coherence_match(query, f_len, j, i, *A_f[l], s);
                float d_app = weighted_features(A_f[l]->ptr()+p_app*f_len, query, gauss_weight);
                float d_coh = weighted_features(A_f[l]->ptr()+p_coh*f_len, query, gauss_weight);
                int p;
                if (d_coh <= w * d_app) p = p_coh;
                else p = p_app;
                if (j < 1 || i < 1 || j > border_y - 2 || i > border_x - 2) p = p_app;
                //cout << j << " " << i << endl;
                s[j*border_x+i] = p;
                int p_x = p % A_pyramid[l].cols, p_y = p / A_pyramid[l].cols;
                // TODO.
                BP_pyramid[l].at<float>(j,i) = AP_pyramid[l].at<float>(p_y, p_x);
                //cout << BP_pyramid[l].at<float>(j,i) - B_pyramid[l].at<float>(j,i)<<endl;
                Bl.at<Vec3b>(j,i) = AP_src[l].at<Vec3b>(p_y, p_x);
            }
            cout << l << ": "<< j << endl;
        }
        
        delete[] indices.ptr();
        delete[] dists.ptr();
        delete[] query;
        delete[] s;
        
        // Restore the original image.
        Mat imgShow;
        img.copyTo(imgShow);
        for (int ll = LEVEL-1; ll > l; ll--) {
            pyrDown(imgShow, imgShow, Size( imgShow.cols/2, imgShow.rows/2 ));
        }
        vector<Mat> channels(3);

        split(imgShow, channels);
        BP_pyramid[l].convertTo(channels[0], CV_8UC1);
        Mat result;
        merge(channels, result);
        cvtColor(result, result, CV_YUV2BGR);
        imshow("image", result);
        imshow("image_", Bl);
        while (1) {
            int k = waitKey();
            if (k == 'q') break;
        }
        IplImage ipl;
        if (tocolor) ipl= IplImage(Bl);
        else ipl = IplImage(result);
        string re_path = "./results/"+to_string(l)+b_path;
        const char* cre_path = re_path.c_str();
        cvSaveImage(cre_path, &ipl);
    }
    for (int i = 0; i < LEVEL; i++) {
        delete [] A_f[i]->ptr();
        delete A_f[i];
    }
    return 0;
}

/**
 * @function readme
 */



