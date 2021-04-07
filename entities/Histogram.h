#ifndef HISTOGRAM_CPP
#define HISTOGRAM_CPP

#include <vector>

#include "../utils/Constants.h"

class Histogram
{
private:
    // long long binary_search(float);
    

    std::vector<float>::iterator begin;
    float data_size;
    int bin_num = Constants::DEFAULT_BIN_NUM;


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
        this->data = data;
        float N = data.size();
        float start_x = data[0];
        float end_x = data[N - 1];
        float key_gap = (end_x - start_x) * 1.0 / bin_num;
        float old_freq = 0.0;
        int index = 0;
        for (size_t i = 1; i < bin_num; i++)
        {
            while(index < N && data[index] <= (start_x + i * key_gap))
            {
                index++;
            }
            hist.push_back(index * 1.0 / N);
        }
        hist.push_back(1.0);
        // if (N < bin_num)
        // {
        //     for (size_t i = 1; i < bin_num; i++)
        //     {
        //         int num = 0;
        //         for (size_t j = 0; j < N; j++)
        //         {
        //             if (data[j] > start_x + key_gap * i && data[j] <= start_x + key_gap * (i + 1))
        //             {
        //                 num++;
        //             }
        //             if (data[j] > start_x + key_gap * (i + 1))
        //             {
        //                 break;
        //             }
                    
        //         }
        //         hist.push_back(float(num) / N);
        //     }
        // }
        // else
        // {
        //     // cout<< "key_gap: " << key_gap << endl;
        //     for (size_t i = 1; i < bin_num; i++)
        //     {
        //         long long index = binary_search(start_x + key_gap * i);
        //         float freq = index * 1.0 / N - old_freq;
        //         hist.push_back(freq);
        //         old_freq = index * 1.0 / N;
        //         // cout<< "i: " << i << " freq:" << freq << endl;
        //     }
        //     // cout<< "finish:" << endl;
        //     hist.push_back(1 - old_freq);
        // }
    }

    // Histogram(std::vector<float>);
    // float cal_dist(std::vector<float>);
    float cal_dist(std::vector<float> source_hist)
    {
        float dist_hist = std::max(source_hist[0], hist[0]);
        float temp_sum = 0.0;

        for (size_t i = 0; i < source_hist.size() - 1; i++)
        {
            temp_sum += source_hist[i] - hist[i];
            float temp = 0.0;
            if (source_hist[i + 1] + temp_sum > hist[i + 1] - temp_sum)
            {
                temp = source_hist[i + 1] + temp_sum;
            }
            else
            {
                temp = hist[i + 1] - temp_sum;
            }
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
        while(data[mid] != key) {
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