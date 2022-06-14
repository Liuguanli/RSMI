#include "Mbr.h"
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <limits>
#include <algorithm>
#include <math.h>
using namespace std;
Mbr::Mbr()
{
}

Mbr::Mbr(float x1, float y1, float x2, float y2)
{
    this->x1 = x1;
    this->y1 = y1;
    this->x2 = x2;
    this->y2 = y2;
}

void Mbr::update(Point point)
{
    update(point.x, point.y);
}

void Mbr::update(float x, float y)
{
    if (x < x1)
    {
        x1 = x;
    }
    if (y < y1)
    {
        y1 = y;
    }
    if (x > x2)
    {
        x2 = x;
    }
    if (y > y2)
    {
        y2 = y;
    }
}

void Mbr::update(Mbr mbr)
{
    if (mbr.x1 < x1)
    {
        x1 = mbr.x1;
    }
    if (mbr.y1 < y1)
    {
        y1 = mbr.y1;
    }
    if (mbr.x2 > x2)
    {
        x2 = mbr.x2;
    }
    if (mbr.y2 > y2)
    {
        y2 = mbr.y2;
    }
}

bool Mbr::contains(Point point)
{
    if (x1 > point.x || point.x > x2 || y1 > point.y || point.y > y2)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool Mbr::strict_contains(Point point)
{
    if (x1 < point.x && point.x < x2 && y1 < point.y && point.y < y2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Mbr::interact(Mbr mbr)
{

    if (x2 < mbr.x1 || mbr.x2 < x1) {
        return false;
    }

    if (y2 < mbr.y1 || mbr.y2 < y1) {
        return false;
    }

    return true;


    // if ((x1 <= mbr.x1 && mbr.x1 <= x2 && y1 <= mbr.y1 && mbr.y1 <= y2) || (x1 <= mbr.x1 && mbr.x1 <= x2 && y1 <= mbr.y2 && mbr.y2 <= y2) || (x1 <= mbr.x2 && mbr.x2 <= x2 && y1 <= mbr.y1 && mbr.y1 <= y2) || (x1 <= mbr.x2 && mbr.x2 <= x2 && y1 <= mbr.y2 && mbr.y2 <= y2))
    // {
    //     return true;
    // }
    // if ((mbr.x1 <= x1 && x1 <= mbr.x2 && mbr.y1 <= y1 && y1 <= mbr.y2) || (mbr.x1 <= x1 && x1 <= mbr.x2 && mbr.y1 <= y2 && y2 <= mbr.y2) || (mbr.x1 <= x2 && x2 <= mbr.x2 && mbr.y1 <= y1 && y1 <= mbr.y2) || (mbr.x1 <= x2 && x2 <= mbr.x2 && mbr.y1 <= y2 && y2 <= mbr.y2))
    // {
    //     return true;
    // }
    // return false;
}

vector<Mbr> Mbr::get_mbrs(vector<Point> dataset, float area, int num, float ratio)
{

    vector<Mbr> mbrs;
    srand(time(0));
    int maxInt = numeric_limits<int>::max();
    float x = sqrt(area * ratio);
    float y = sqrt(area / ratio);
    int i = 0;
    int length = dataset.size();
    while (i < num)
    {
        int index = rand() % length;
        Point point = dataset[index];
        if (point.x + x <= 1 && point.y + y <= 1)
        {
            Mbr mbr(point.x, point.y, point.x + x, point.y + y);
            mbrs.push_back(mbr);
            i++;
        }
    }

    return mbrs;
}

float Mbr::cal_dist(Point point)
{
    if (this->contains(point))
    {
        return 0;
    }
    else
    {
        float dist;
        if (point.x < x1)
        {
            if (point.y < y1)
            {
                dist = sqrt(pow((point.x - x1), 2) + pow((point.y - y1), 2));
            }
            else if (point.y <= y2)
            {
                dist = x1 - point.x;
            }
            else
            {
                dist = sqrt(pow((point.x - x1), 2) + pow((point.y - y2), 2));
            }
        }
        else if (point.x <= x2)
        {
            if (point.y < y1)
            {
                dist = y1 - point.y;
            }
            else
            {
                dist = point.y - y2;
            }
        }
        else
        {
            if (point.y < y1)
            {
                dist = sqrt(pow((point.x - x2), 2) + pow((point.y - y1), 2));
            }
            else if (point.y <= y2)
            {
                dist = point.x - x2;
            }
            else
            {
                dist = sqrt(pow((point.x - x2), 2) + pow((point.y - y2), 2));
            }
        }
        return dist;
    }
}

void Mbr::print()
{
    cout << "(x1=" << x1 << " y1=" << y1 << " x2=" << x2 << " y2=" << y2 << ")" << endl;
}

vector<Point> Mbr::get_corner_points()
{
    vector<Point> result;
    Point point1(x1, y1);
    Point point2(x2, y1);
    Point point3(x1, y2);
    Point point4(x2, y2);
    result.push_back(point1);
    result.push_back(point2);
    result.push_back(point3);
    result.push_back(point4);
    return result;
}

Mbr Mbr::get_mbr(Point point, float knnquerySide)
{
    float x1 = point.x - knnquerySide;
    float x2 = point.x + knnquerySide;
    float y1 = point.y - knnquerySide;
    float y2 = point.y + knnquerySide;

    x1 = x1 < 0 ? 0 : x1;
    y1 = y1 < 0 ? 0 : y1;

    x2 = x2 > 1 ? 1 : x2;
    y2 = y2 > 1 ? 1 : y2;

    Mbr mbr(x1, y1, x2, y2);
    return mbr;
}

void Mbr::clean()
{
    x1 = 0;
    x2 = 0;
    y1 = 0;
    y2 = 0;
}

string Mbr::get_self()
{
    return to_string(x1) + "," + to_string(y1) + "," + to_string(x2) + "," + to_string(y2) + "\n";
}