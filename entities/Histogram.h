#ifndef HISTOGRAM_CPP
#define HISTOGRAM_CPP

#include <vector>

#include "../utils/Constants.h"
#include <cmath>

class Histogram
{
private:
    // long long binary_search(float);

    std::vector<float>::iterator begin;
    float data_size;
    int bin_num = Constants::DEFAULT_BIN_NUM;

    float start_x = 0;
    float end_x = 0;
    float key_gap = 0;

    long N = 0;

public:
    std::vector<float> hist;
    std::vector<float> data;

    Histogram()
    {
    }


    Histogram(int bin_num)
    {
        this->bin_num = bin_num;
    }

    Histogram(int bin_num, std::vector<float> data)
    {
        // this->data = data;
        N = data.size();
        start_x = data[0];
        end_x = data[N - 1];

        bin_num = 100;

        this->bin_num = bin_num;
        key_gap = (end_x - start_x) * 1.0 / bin_num;
        float old_freq = 0.0;
        int index = 0;
        for (size_t i = 1; i < bin_num; i++)
        {
            while (index < N && data[index] <= (start_x + i * key_gap))
            {
                index++;
            }
            hist.push_back(index * 1.0 / N);
        }
        hist.push_back(1.0);
    }

    void update(float value)
    {
        int index = ceil((value - start_x) * 1.0 / key_gap);
        for (size_t i = index; i < hist.size(); i++)
        {
            hist[i] = hist[i] + 1.0 / N;
        }
        for (size_t i = 0; i < hist.size(); i++)
        {
            hist[i] = hist[i] * N / (N + 1);
        }
        N++;
    }

    // Histogram(std::vector<float>);
    // float cal_dist(std::vector<float>);
    float cal_cdf_dist(std::vector<float> source_hist)
    {
        float dist_hist = 0;
        for (size_t i = 0; i < source_hist.size(); i++)
        {
            float temp = abs(source_hist[i] - hist[i]);
            
            if (dist_hist < temp)
            {
                dist_hist = temp;
            }
        }
        return dist_hist;
    }

    double cal_similarity(vector<float> target_hist)
    {
        double max_gap = 0.0;
        int length = hist.size();
        // assert(length == histogram.hist.size());
        for (size_t i = 0; i < length; i++)
        {
            double temp_gap = abs(hist[i] - target_hist[i]);
            if (max_gap < temp_gap)
            {
                max_gap = temp_gap;
            }
        }
        return max_gap;
    }

    long long binary_search(float key)
    {
        long long begin = 0;
        long long end = data.size() - 1;
        long long mid = (begin + end) / 2;
        while (data[mid] != key)
        {
            if (data[mid] < key)
            {
                if (data[mid + 1] >= key)
                {
                    break;
                }
                begin = mid;
            }
            else if (data[mid] > key)
            {
                if (data[mid - 1] <= key)
                {
                    mid -= 1;
                    break;
                }
                end = mid;
            }
            mid = (begin + end) / 2;
        }
        return mid;
    }
};

#endif