#ifndef SORTTOOLS_H
#define SORTTOOLS_H

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include "../utils/Constants.h"
#include "../entities/Mbr.h"
#include "../entities/Point.h"
#include "../entities/NodeExtend.h"

using namespace std;

// long long diff(struct timespec start, struct timespec end)
// {
//     struct timespec temp;
//     long long gap;
//     if ((end.tv_nsec - start.tv_nsec) < 0)
//     {
//         temp.tv_sec = end.tv_sec - start.tv_sec - 1;
//         temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
//     }
//     else
//     {
//         temp.tv_sec = end.tv_sec - start.tv_sec;
//         temp.tv_nsec = end.tv_nsec - start.tv_nsec;
//     }
//     gap = temp.tv_sec * 1000000000 + temp.tv_nsec;
//     return gap;
// }

// struct timespec diff(struct timespec start, struct timespec end)
// {
//     struct timespec temp;
//     if ((end.tv_nsec - start.tv_nsec) < 0)
//     {
//         temp.tv_sec = end.tv_sec - start.tv_sec - 1;
//         temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
//     }
//     else
//     {
//         temp.tv_sec = end.tv_sec - start.tv_sec;
//         temp.tv_nsec = end.tv_nsec - start.tv_nsec;
//     }
//     return temp;
// }

struct sortPQ
{
    bool operator()(const NodeExtend *node1, const NodeExtend *node2)
    {
        return (node1->dist > node2->dist);
    }
};

struct sortPQ_Desc
{
    bool operator()(const NodeExtend *node1, const NodeExtend *node2)
    {
        return (node1->dist < node2->dist);
    }
};

struct sortForKNN
{
    Point queryPoint;
    sortForKNN(Point &point)
    {
        queryPoint = point;
    }
    bool operator()(Point point1, Point point2)
    {
        return (point1.cal_dist(queryPoint) < point2.cal_dist(queryPoint));
    }
};

struct sortForKNN1
{
    bool operator()(Point point1, Point point2)
    {
        return point1.temp_dist < point2.temp_dist;
    }
};

struct sortForKNN2
{
    bool operator()(Point point1, Point point2)
    {
        return point1.temp_dist > point2.temp_dist;
    }
};

struct sortX
{
    bool operator()(const Point point1, const Point point2)
    {
        return (point1.x < point2.x);
    }
};

struct sortY
{
    bool operator()(const Point point1, const Point point2)
    {
        return (point1.y < point2.y);
    }
};

struct sort_curve_val
{
    bool operator()(const Point point1, const Point point2)
    {
        return (point1.curve_val < point2.curve_val);
    }
};

#endif