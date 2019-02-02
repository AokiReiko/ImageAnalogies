//
//  ann_match.hpp
//  ImageAnalogies
//
//  Created by aokireiko on 18/12/3.
//  Copyright © 2018年 aokireiko. All rights reserved.
//

#ifndef ann_match_hpp
#define ann_match_hpp

#include <stdio.h>
#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

using namespace flann;
using namespace std;


class AnnMatch {
    Index<L2<float>> *ann_index;
public:
    AnnMatch(Matrix<float>& data, int point_num, int f_len);
    ~AnnMatch();
    void best_ann_match(Matrix<int>& indices, Matrix<float>& dists, float* query, int len);
};

#endif /* ann_match_hpp */
