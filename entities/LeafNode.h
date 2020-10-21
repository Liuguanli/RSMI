#ifndef LEAFNODE_H
#define LEAFNODE_H

#include <vector>
#include "Node.h"
#include "Point.h"
#include "Mbr.h"
#include "NonLeafNode.h"
using namespace std;

class LeafNode : public nodespace::Node
{
public:
    int level;
    vector<Point> *children;
    NonLeafNode *parent;
    LeafNode();
    LeafNode(Mbr mbr);
    void add_point(Point);
    void add_points(vector<Point>);
    bool delete_point(Point);
    bool is_full();
    LeafNode *split();
    LeafNode split1();
};

#endif