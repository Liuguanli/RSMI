#include "NonLeafNode.h"
#include "LeafNode.h"
#include "../utils/Constants.h"
#include <vector>
#include <iostream>
#include <typeinfo>
using namespace std;

NonLeafNode::NonLeafNode()
{
    children = new vector<Node *>();
}

NonLeafNode::NonLeafNode(Mbr mbr)
{
    children = new vector<Node *>();
    this->mbr = mbr;
}

void NonLeafNode::addNode(Node *node)
{
    //assert(children->size() <= Constants::PAGESIZE);
    // add
    children->push_back(node);
    // update MBR
    mbr.update(node->mbr);

    if (typeid(*node) == typeid(NonLeafNode))
    {
        NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(node);
        nonLeafNode->parent = this;
    }
    else
    {
        LeafNode *leafNode = dynamic_cast<LeafNode *>(node);
        leafNode->parent = this;
    }
}

void NonLeafNode::addNodes(vector<Node *> nodes)
{
    for (int i = 0; i < nodes.size(); i++)
    {
        addNode(nodes[i]);
    }
    // cout<< mbr.x1 << " " << mbr.y1 << " " << mbr.x2 << " " << mbr.y2 << endl;
}

bool NonLeafNode::is_full()
{
    return children->size() >= Constants::PAGESIZE;
}

NonLeafNode *NonLeafNode::split()
{
    // build rightNode
    NonLeafNode *right = new NonLeafNode();
    right->parent = this->parent;
    int mid = Constants::PAGESIZE / 2;
    auto bn = children->begin() + mid;
    auto en = children->end();
    vector<Node *> vec(bn, en);
    right->addNodes(vec);
    
    // build leftNode
    auto bn1 = children->begin();
    auto en1 = children->begin() + mid;
    vector<Node *> vec1(bn1, en1);
    children->clear();
    addNodes(vec1);

    return right;
}
