#include "Point.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <string.h>
#include <algorithm>
#include <math.h>
using namespace std;

Point::Point()
{
}

Point::Point(float x, float y)
{
    this->x = x;
    this->y = y;
}

bool Point::operator==(const Point &point)
{
    if (this == &point)
    {
        return true;
    }
    else if (this->x == point.x && this->y == point.y)
    {
        return true;
    }
    return false;
}

double Point::cal_dist(Point point)
{
    // if (temp_dist == 0)
    temp_dist = sqrt(pow((point.x - x), 2) + pow((point.y - y), 2));
    return temp_dist;
    // return sqrt(pow((point.x - x), 2) + pow((point.y - y), 2));
}

void Point::print()
{
    cout << "(x=" << x << ",y=" << y << ")"
         << " index=" << index << " curve_val=" << curve_val << endl;
}

vector<Point> Point::get_points(vector<Point> dataset, int num)
{
    cout<< "..." << dataset.size() << endl;
    srand(time(0));
    int length = dataset.size();
    vector<Point> points;
    for (int i = 0; i < num; i++)
    {
        int index = rand() % length;
        points.push_back(dataset[index]);
    }
    return points;
}

vector<Point> Point::get_inserted_points(long long num, string distribution)
{
    srand(time(0)); // initialize random number generator which always be a different value
    vector<Point> points;
    for (int i = 0; i < num; i++)
    {
        float x = (float)(rand() % num) / num;
        float y = 0;
        y = (float)(rand() % num) / num;
        if (distribution == "skew")
        {
            // y = (float)(rand() % num) / num;
            y = pow(y, 4);
        }
        Point point(x, y);
        points.push_back(point);
    }
    return points;
}

string Point::get_self()
{
    return to_string(x) + "," + to_string(y) + "\n";
}
