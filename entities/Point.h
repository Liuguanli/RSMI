#ifndef POINT_H
#define POINT_H
#include <vector>
#include <string.h>
#include <string>
using namespace std;
class Point
{

public:
    double index = 0.0;
    float x = 0.0;
    float y = 0.0;
    long long x_i = 0;
    long long y_i = 0;
    // long long xs[2];
    long long curve_val = 0;
    float normalized_curve_val = 0.0;
    double ml_normalized_curve_val = 0.0;

    double temp_dist = 0.0;

    int partition_id = 0;
    double key = 0;

    bool is_deleted = false;

    Point(float, float);
    Point();
    bool operator==(const Point &point);
    double cal_dist(Point);
    void print();
    static vector<Point> get_points(vector<Point>, int);
    static vector<Point> get_inserted_points(long long, string distribution);

    string get_self();
};

#endif
