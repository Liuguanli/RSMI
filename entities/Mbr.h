#ifndef MBR_H
#define MBR_H
#include <limits>
#include <vector>
#include "Point.h"
using namespace std;

class Mbr
{
public:
    float x1 = numeric_limits<float>::max();
    float x2 = numeric_limits<float>::min();
    float y1 = numeric_limits<float>::max();
    float y2 = numeric_limits<float>::min();
    Mbr();
    Mbr(float, float, float, float);
    void update(float, float);
    void update(Point);
    void update(Mbr);
    bool contains(Point);
    bool strict_contains(Point);
    bool interact(Mbr);
    static vector<Mbr> get_mbrs(vector<Point>, float, int, float);
    float cal_dist(Point);
    void print();
    vector<Point> get_corner_points();
    static Mbr get_mbr(Point point, float knnquerySide);
    void clean();
    string get_self();
};

#endif