//
//  ann_match.cpp
//  ImageAnalogies
//
//  Created by aokireiko on 18/12/3.
//  Copyright © 2018年 aokireiko. All rights reserved.
//

#include "ann_match.hpp"
AnnMatch::AnnMatch(Matrix<float>& data, int point_num, int f_len) {
    ann_index = new Index<L2<float>>(data, KDTreeIndexParams(4));
    ann_index->buildIndex();
    
}

AnnMatch::~AnnMatch() {
    delete ann_index;
    
}

void AnnMatch::best_ann_match(Matrix<int>& indices, Matrix<float>& dists, float* query, int len) {
    Matrix<float> q(query, 1, len);
    ann_index->knnSearch(q, indices, dists, 1, SearchParams(128));
}