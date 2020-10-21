#ifndef NONLEAFNODE_H
#define NONLEAFNODE_H

#include <vector>
#include "Node.h"

class NonLeafNode : public nodespace::Node
{
public:
    int level;
    vector<Node*> *children;
    NonLeafNode *parent;
    NonLeafNode();
    NonLeafNode(Mbr mbr);
    void addNode(Node*);
    void addNodes(vector<Node*>);
    bool is_full();
    NonLeafNode* split();
};

#endif