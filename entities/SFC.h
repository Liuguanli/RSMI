#ifndef SFC_CPP
#define SFC_CPP

#include <vector>

#include "../utils/Constants.h"

class SFC
{
private:
    // int bit_num;
    // std::vector<long long> values;
    // std::vector<double> cdf;

public:

    int bit_num;
    std::vector<long long> values;
    std::vector<double> cdf;
    std::vector<double> pdf;

    SFC()
    {

    }

    SFC(int bit_num, std::vector<long long> values)
    {
        this->bit_num = bit_num;
        this->values = values;
    }


    // SFC shrink(int to_bit_num)
    // {
    //     if (cdf.size() == 0)
    //     {
    //         gen_CDF();
    //         gen_PDF();
    //     }
    //     // assert(cdf.size() > 0);
    //     // assert(to_bit_num < this->bit_num);
    //     int time = pow(2, (bit_num - to_bit_num));
    //     SFC sfc;
    //     long long max = pow(2, bit_num);
    //     std::vector<double> shrinked_cdf;
    //     std::vector<double> shrinked_pdf;
    //     for (size_t i = 0; i < max; i += time)
    //     {
    //         double sum = 0;
    //         for (size_t j = 0; j < time; j++)
    //         {
    //             sum += pdf[i + j];
    //         }
    //         shrinked_pdf.push_back(sum);
    //     }

    //     for (size_t i = 0; i < max; i += time)
    //     {
    //         shrinked_cdf.push_back(cdf[i + time - 1]);
    //     }
    //     sfc.bit_num = to_bit_num;
    //     sfc.cdf = shrinked_cdf;
    //     sfc.pdf = shrinked_pdf;
    //     return sfc;
    // }

    vector<int> get_counted_courve()
    {
        long long max = pow(2, bit_num) - 1;
        vector<int> counted_curve_value;
        int N = values.size();
        int index = 0;
        for (size_t i = 0; i <= max; i++)
        {
            int each_index = 0;
            while(values[index] == i && index < N)
            {
                index++;
                each_index++;
            }
            counted_curve_value.push_back(each_index);
        }
        long long sum = 0;
        for (size_t i = 0; i <= max; i++)
        {
            sum += counted_curve_value[i];
        }
        // cout<< "counted sum: " << sum << endl;
        return counted_curve_value;
    }

    vector<float> get_weighted_curve()
    {
        long long max = pow(2, bit_num) - 1;
        vector<float> weighted_curve_value;
        int N = values.size();
        int index = 0;
        for (size_t i = 0; i <= max; i++)
        {
            int each_index = 0;
            while(values[index] == i && index < N)
            {
                index++;
                each_index++;
            }
            weighted_curve_value.push_back(each_index * 1.0 / N);
        }
        float sum = 0;
        for (size_t i = 0; i <= max; i++)
        {
            sum += weighted_curve_value[i];
        }
        // cout<< "weighted sum: " << sum << endl;
        return weighted_curve_value;
    }

    vector<float> get_weighted_curve(int to_bit_num)
    {
        long long max = pow(2, to_bit_num) - 1;
        vector<float> weighted_curve_value;
        int N = values.size();
        long long gap = pow(2, bit_num - to_bit_num);
        for (size_t i = 0; i <= max; i++)
        {

            int index = 0;
            while(values[index] / gap == i && index < N)
            {
                index++;
            }
            weighted_curve_value.push_back(index * 1.0 / N);
        }
        return weighted_curve_value;
    }

    void gen_CDF()
    {
        long long max = pow(2, bit_num) - 1;
        int N = values.size();
        int index = 0;
        for (size_t i = 0; i <= max; i++)
        {
            while(values[index] == i)
            {
                index++;
            }
            cdf.push_back(index * 1.0 / N);
        }
    }

    void gen_CDF(int to_bit_num)
    {
        long long max = pow(2, to_bit_num) - 1;
        int N = values.size();
        int index = 0;
        long long gap = pow(2, bit_num - to_bit_num);
        for (size_t i = 0; i <= max; i++)
        {
            while(values[index] / gap == i && index < N)
            {
                index++;
            }
            cdf.push_back(index * 1.0 / N);
        }
    }

    // void gen_PDF()
    // {
    //     long long max = (pow(2, bit_num) - 1);
    //     int N = values.size();
    //     int index = 0;
    //     int old_index = 0;
    //     for (size_t i = 0; i <= max; i++)
    //     {
    //         while(values[index] == i)
    //         {
    //             index++;
    //         }
    //         pdf.push_back((index - old_index) * 1.0 / N);
    //         old_index = index;
    //     }
    // }

    double cal_similarity(Histogram histogram)
    {
        if (cdf.size() == 0)
        {
            this->gen_CDF();
        }
        double max_gap = 0.0;
        int length = cdf.size();

        assert(length == histogram.hist.size());

        for (size_t i = 0; i < length; i++)
        {
            double temp_gap = abs(cdf[i] - histogram.hist[i]);
            if (max_gap < temp_gap)
            {
                max_gap = temp_gap;
            }
        }
        return max_gap;
    }

    double cal_similarity(SFC sfc)
    {
        if (cdf.size() == 0)
        {
            this->gen_CDF();
        }
        if (sfc.cdf.size() == 0)
        {
            sfc.gen_CDF();
        }
        double max_gap = 0.0;
        int length = cdf.size();

        assert(length == sfc.cdf.size());

        for (size_t i = 0; i < length; i++)
        {
            double temp_gap = abs(cdf[i] - sfc.cdf[i]);
            if (max_gap < temp_gap)
            {
                max_gap = temp_gap;
            }
        }
        return max_gap;
    }
};

#endif