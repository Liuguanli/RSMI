#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
// #include <boost/algorithm/string.hpp>
using namespace std;

class SearchHelper
{
public:
    template <typename T>
    static long binarySearch(vector<T>, T);
};

template <typename T>
long SearchHelper::binarySearch(vector<T> values, T target)
{
    long begin = 0;
    long end = values.size() - 1;
    if (target <= values[begin])
    {
        return begin;
    }
    if (target >= values[end])
    {
        return end;
    }
    long mid = (begin + end) / 2;
    T current = values[mid];
    while (values[mid] > target || values[mid + 1] < target)
    {
        if (values[mid] > target)
        {
            end = mid;
        }
        else if (values[mid] < target)
        {
            begin = mid;
        }
        else
        {
            return mid;
        }
        mid = (begin + end) / 2;
        current = values[mid];
    }
    return mid;
}