#ifndef APPROXSFC_CPP
#define APPROXSFC_CPP

#include <vector>

#include "../utils/Constants.h"

class ApproxSFC
{
private:
    int bit_nums[] = {6, 8, 10, 12, 14, 16, 18, 20};

    int bit_num;
    std::vector<long long> values;
    map<int, std::vector<double>> approx_values;

public:
    ApproxSFC(int bit_num, std::vector<long long> values)
    {
        this->bit_num = bit_num;
        this->values = values;

        int bit_nums_len = sizeof(bit_nums) / sizeof(bit_nums[0]);
        for (size_t i = 0; i < bit_nums_len; i++)
        {
            std::vector<double> cdf;
            long long old_value = values[0] / pow(2, bit_num - bit_nums[i]);
            int num = 1;
            for (size_t j = 1; j < values.size(); j++)
            {
                long long value = values[j];
                value = value / pow(2, bit_num - bit_nums[i]);
                if (old_value == value)
                {
                    num++;
                }
                else
                {
                    num = 1;
                    old_value = value;
                    cdf.push_back(num * 1.0 / (pow(2, bit_num) - 1));
                }
            }
            cout<< "bit_nums[i]: " << bit_nums[i] << endl;
            cout<< "cdf.size(): " << cdf.size() << endl;
            approx_values.insert(pair<int, std::vector<double>>(bit_nums[i], cdf));
        }
    }

    double cal_similarity(int bit_num, ApproxSFC another)
    {
        double res = 0;
        long num = pow(2, bit_num);
    }
};

#endif