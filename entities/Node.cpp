#include "Node.h"

nodespace::Node::Node()
{
}

float nodespace::Node::cal_dist(Point point)
{
    return mbr.cal_dist(point);
}