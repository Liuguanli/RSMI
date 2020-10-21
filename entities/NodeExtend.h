#ifndef NODEEXTEND_H
#define NODEEXTEND_H

#include <vector>
#include "Node.h"
#include "Point.h"
// #include "Mbr.h"
// using namespace std;

class NodeExtend{

    public:
        nodespace::Node *node = NULL;
        Point point;
        float dist;
        NodeExtend();
        NodeExtend(nodespace::Node*, float);
        NodeExtend(Point, float);
        bool is_leafnode();
};

#endif