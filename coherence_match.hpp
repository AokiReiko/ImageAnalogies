//
//  coherence_match.hpp
//  ImageAnalogies
//
//  Created by aokireiko on 18/12/3.
//  Copyright © 2018年 aokireiko. All rights reserved.
//

#ifndef coherence_match_hpp
#define coherence_match_hpp

#include <stdio.h>
#include <map>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

using namespace flann;
using namespace std;

class CoherenceMatch {
private:
    int cols_A, rows_A;
    int cols_B, rows_B;
public:
    CoherenceMatch(int ca, int ra, int cb, int rb):
        cols_A(ca),
        rows_A(ra),
        cols_B(cb),
        rows_B(rb)
    {
        printf("CoherenceMatch: A(%d,%d), B(%d,%d)\n", cols_A, rows_A, cols_B, rows_B);
    }
    int best_coherence_match(float* query, int len, int j, int i, Matrix<float>& A_features, int* s);
private:
    float compute_dist(const float* a, const float* b, int len);
    
};

#endif /* coherence_match_hpp */
