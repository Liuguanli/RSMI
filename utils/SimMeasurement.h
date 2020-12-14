#include <string>
#include <iostream>
#include <fstream>
#include "../entities/Feature.h"

using namespace std;

namespace similarity_utils
{
    inline double sim_cal(Feature source, Feature target);
};

namespace similarity_utils
{
    double sim_cal(Feature source, Feature target)
    {
        return 1.0;
    }
}