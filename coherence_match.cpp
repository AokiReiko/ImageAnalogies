
//
//  coherence_match.cpp
//  ImageAnalogies
//
//  Created by aokireiko on 18/12/3.
//  Copyright © 2018年 aokireiko. All rights reserved.
//

#include "coherence_match.hpp"


// Return s(r*) + (q-r*)
int CoherenceMatch::best_coherence_match(float* query, int len, int y, int x, Matrix<float>& A_features, int* s) {
    int ind = 0;
    float min_dist = (float) INT32_MAX;
    int low_y = y > 2 ? y-2 : 0;
    int left_x = x > 2 ? x-2 : 0, right_x = x+2 < cols_B ? x+2: cols_B-1;
    
    for (int j = low_y; j < y; j++) {
        for (int i = left_x; i <= right_x; i++) {
            //assert(s.find(j*cols_B+i) != s.end());
            int s_r = s[j*cols_B+i];
            int r_x = s_r % cols_A, r_y = s_r / cols_A;
            int temp = r_x - i + x;
            if (temp > 0 && temp < cols_A) r_x = temp;
            temp = r_y - j + y;
            if (temp > 0 && temp < rows_A) r_y = temp;
            s_r = r_y * cols_A + r_x;
            
            float dist = compute_dist(query, A_features.ptr()+len*s_r, len);
            if (dist < min_dist) {
                min_dist = dist;
                ind = s_r;
            }
        }
    }
    for (int i = left_x; i < x; i++) {
        //assert(s.count(y*cols_B+i));
        int s_r = s[y*cols_B+i];
        int r_x = s_r % cols_A, r_y = s_r / cols_A;
        int temp = r_x - i + x;
        if (temp > 0 && temp < cols_A) r_x = temp;
        s_r = r_x+r_y*cols_A;
        float dist = compute_dist(query, A_features.ptr()+len*s_r, len);
        if (dist < min_dist) {
            min_dist = dist;
            ind = s_r;
        }
    }
    
    return ind;
}

float CoherenceMatch::compute_dist(const float* a, const float *b, int len) {
    float res = 0;
    for (int i = 0; i < len; i++) {
        float sub = a[i]-b[i];
        res += sub*sub;
    }
    
    return res;
}
