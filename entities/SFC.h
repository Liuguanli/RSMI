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
            while (values[index] == i && index < N)
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
        // TODO use map to get
        map<long long, int> values_map;
        for (size_t i = 0; i < N; i++)
        {
            values_map[values[i]]++;
        }
        vector<long long> keys;
        int keys_size = values_map.size();
        keys.reserve(keys_size);
        for (const auto &[key, value] : values_map)
        {
            keys.push_back(key);
        }
        sort(keys.begin(), keys.end());
        int temp_sum = 0;
        for (size_t i = 0; i < keys_size; i++)
        {
            // temp_sum += values_map[keys[i]];
            weighted_curve_value.push_back(values_map[keys[i]] * 1.0 / N);
        }
        cout << "weighted_curve_value: " << weighted_curve_value << endl;
        float sum = 0;
        for (size_t i = 0; i <= max; i++)
        {
            sum += weighted_curve_value[i];
        }
        cout << "weighted sum: " << sum << endl;
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
            while (values[index] / gap == i && index < N)
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
            while (values[index] == i)
            {
                index++;
            }
            cdf.push_back(index * 1.0 / N);
        }
    }

    //TODO to be tested
    void gen_CDF(int to_bit_num)
    {
        long long max = pow(2, to_bit_num) - 1;
        int N = values.size();
        int index = 0;
        long long gap = pow(2, bit_num - to_bit_num);
        cout << "gap1: " << gap << endl;
        for (size_t i = 0; i <= max; i++)
        {
            while (values[index] / gap == i && index < N)
            {
                index++;
            }
            cdf.push_back(index * 1.0 / N);
        }
        cout << "cdf: " << cdf << endl;
        int start_index = max - 1;
        double start = cdf[start_index];
        for (size_t i = max - 2; i >= 0; i++)
        {
            if (cdf[i] == start)
            {
                continue;
            }
            else
            {
                if ((start_index - i) > 1)
                {
                    double gap = (start - cdf[i]) / (start_index - i);
                    for (size_t j = 1; j < start_index - i; j++)
                    {
                        cdf[i + j] += gap * j;
                    }
                }
                start_index = i;
                start = cdf[start_index];
            }
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

    // TODO make sure sfc
    // TODO then different shapes of SFC
    // TODO Consider h-curve
    // vector<vector<int>> cells4_sfcs;
    // vector<vector<int>> synthetic_sfcs;
    // void enumerate_4cells_sfc(vector<int> sfc, int b0_num, int b1_num, int length)
    // {
    //     if (sfc.size() == length)
    //     {
    //         cout<< "sfc: " << sfc << endl;
    //         cells4_sfcs.push_back(sfc);
    //         return;
    //     }
    //     if (b0_num > 0)
    //     {
    //         vector<int> sfc_copy(sfc);
    //         sfc_copy.push_back(0);
    //         enumerate_4cells_sfc(sfc_copy, b0_num - 1, b1_num, length);
    //     }
    //     if (b1_num > 0)
    //     {
    //         vector<int> sfc_copy(sfc);
    //         sfc_copy.push_back(1);
    //         enumerate_4cells_sfc(sfc_copy, b0_num, b1_num - 1, length);
    //     }
    // }

    // void enumerate_synthetic_sfc(vector<int> sfc, int length)
    // {
    //     if (length == 4)
    //     {
    //         synthetic_sfcs.push_back(sfc);
    //         cout<< "sfc: " << sfc << endl;
    //         return;
    //     }
    //     for (size_t i = 0; i < cells4_sfcs.size(); i++)
    //     {
    //         vector<int> sfc_copy(sfc);
    //         sfc_copy.insert(sfc_copy.end(), cells4_sfcs[i].begin(), cells4_sfcs[i].end());
    //         enumerate_synthetic_sfc(sfc_copy, length + 1);
    //     }
    // }

    // void gen_4cells_SFC()
    // {
    //     int total_points[] = {1, 2, 3, 4};
    //     vector<int> sfc;
    //     for (size_t i = 0; i < sizeof(total_points) / sizeof(total_points[0]); i++)
    //     {
    //         enumerate_4cells_sfc(sfc, 4 - total_points[i], total_points[i], 4);
    //     }
    // }

    // void gen_SFC()
    // {
    //     vector<int> sfc;
    //     enumerate_synthetic_sfc(sfc, 0);
    //     cout<< "synthetic_sfcs size: " << synthetic_sfcs.size() << endl;
    // }
};

#endif