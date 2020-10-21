#include "NodeExtend.h"
#include <iostream>
#include "NonLeafNode.h"
#include <typeinfo>
using namespace std;

NodeExtend::NodeExtend()
{
}

NodeExtend::NodeExtend(Point point, float dist)
{
    this->point = point;
    this->dist = dist;
}

NodeExtend::NodeExtend(nodespace::Node *node, float dist)
{
    this->node = node;
    this->dist = dist;
}

bool NodeExtend::is_leafnode()
{
    if (typeid(*node) == typeid(NonLeafNode))
    {
        return false;
    }
    return true;
}
