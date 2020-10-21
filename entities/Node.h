#ifndef NODE_H
#define NODE_H
#include "Mbr.h"
namespace nodespace
{
class Node
{
public:
    Mbr mbr;
    int order_in_level;
    Node();
    virtual ~Node() {}
    float cal_dist(Point);
};
};
#endif