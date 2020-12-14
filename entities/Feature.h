#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include "Histogram.h"
#include "SFC.h"
using namespace std;

class Feature
{
public:
    
    Histogram histogram;
    SFC sfc;

    Feature()
    {
        
    }


};

#endif