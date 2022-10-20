// #ifndef HRR_H
// #define HRR_H

// #include <iostream>
// #include <fstream>
// #include <boost/algorithm/string.hpp>
// #include <vector>
// #include <iterator>
// #include <string>
// #include <algorithm>

// #include "HRR.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <queue>
#include <iterator>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../utils/ExpRecorder.h"
#include "../curves/hilbert.H"
#include "../curves/hilbert4.H"
#include "../entities/LeafNode.h"
#include "../entities/Node.h"
#include "../entities/NonLeafNode.h"
#include "../utils/Constants.h"
#include "../entities/NodeExtend.h"
#include "../utils/SearchHelper.h"
#include "../utils/SortTools.h"

#include <chrono>

// #include "../utils/ExpRecorder.h"
// #include "../entities/Mbr.h"
// #include "../entities/Point.h"
// #include "../entities/LeafNode.h"
// #include "../entities/NonLeafNode.h"
using namespace std;
class HRR
{
private:
    string fileName;
    int pageSize;
    long long side;

public:
    HRR();
    HRR(int);
    HRR(string, int);

    vector<float> xs;
    vector<float> ys;
    vector<long long> hs;

    NonLeafNode *root;
    void build(ExpRecorder &expRecorder, vector<Point> points);

    void point_query(ExpRecorder &expRecorder, Point queryPoints);
    void point_query(ExpRecorder &expRecorder, vector<Point> queryPoints);

    void window_query(ExpRecorder &expRecorder, vector<Mbr> queryWindows);
    vector<Point> window_query(ExpRecorder &expRecorder, Mbr queryWindows);

    void kNN_query(ExpRecorder &expRecorder, vector<Point> queryPoints, int k);
    vector<Point> kNN_query(ExpRecorder &expRecorder, Point queryPoint, int k);

    void insert(ExpRecorder &expRecorder, Point);
    void insert(ExpRecorder &expRecorder, vector<Point> points);

    void remove(ExpRecorder &expRecorder, Point);
    void remove(ExpRecorder &expRecorder, vector<Point>);
};

// #endif

HRR::HRR()
{
    this->pageSize = Constants::PAGESIZE;
}

HRR::HRR(int pageSize)
{
    this->pageSize = pageSize;
}

HRR::HRR(string fileName, int pageSize)
{
    this->fileName = fileName;
    this->pageSize = pageSize;
}

void HRR::build(ExpRecorder &expRecorder, vector<Point> points)
{
    cout << "HRR::build" << endl;
    auto start = chrono::high_resolution_clock::now();
    // cout << points.size() << endl;
    sort(points.begin(), points.end(), sortX());
    for (long i = 0; i < points.size(); i++)
    {
        points[i].x_i = i;
        xs.push_back(points[i].x);
    }
    sort(points.begin(), points.end(), sortY());
    for (long i = 0; i < points.size(); i++)
    {
        points[i].y_i = i;
        ys.push_back(points[i].y);
    }
    side = pow(2, ceil(log(points.size()) / log(2)));
    for (long i = 0; i < points.size(); i++)
    {
        long long currentVal = compute_Hilbert_value(points[i].x_i, points[i].y_i, side);
        points[i].curve_val = currentVal;
        hs.push_back(currentVal);
    }
    sort(points.begin(), points.end(), sort_curve_val());
    int leaf_node_num = points.size() / pageSize;
    // cout << "leaf_node_num:" << leaf_node_num << endl;
    vector<nodespace::Node *> *leafnodes = new vector<nodespace::Node *>();
    for (int i = 0; i < leaf_node_num; i++)
    {
        LeafNode *leafNode = new LeafNode();
        auto bn = points.begin() + i * pageSize;
        auto en = points.begin() + i * pageSize + pageSize;
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0]->x_i << " " << vec[99]->x_i << endl;
        leafNode->add_points(vec);
        nodespace::Node *temp = leafNode;
        leafnodes->push_back(temp);
    }
    expRecorder.leaf_node_num = leaf_node_num;
    // for the last leafNode
    if (points.size() > pageSize * leaf_node_num)
    {
        // TODO if do not delete will it last to the end of lifecycle?
        LeafNode *leafNode = new LeafNode();
        auto bn = points.begin() + pageSize * leaf_node_num;
        auto en = points.end();
        vector<Point> vec(bn, en);
        // cout << vec.size() << " " << vec[0].x_i << " " << vec[99].x_i << endl;
        leafNode->add_points(vec);
        leafnodes->push_back(leafNode);
        expRecorder.leaf_node_num++;
    }
    int level = ceil(log(points.size()) / log(pageSize)) - 1;
    for (int i = 0; i < level; i++)
    {
        vector<NonLeafNode *> *nonleafnodes = new vector<NonLeafNode *>();
        int nodeNum = leafnodes->size() / pageSize;
        expRecorder.non_leaf_node_num += nodeNum;
        for (int j = 0; j < nodeNum; j++)
        {
            NonLeafNode *nonLeafNode = new NonLeafNode();
            nonLeafNode->level = i + 1;
            auto bn = leafnodes->begin() + j * pageSize;
            auto en = leafnodes->begin() + j * pageSize + pageSize;
            vector<nodespace::Node *> vec(bn, en);
            // cout << vec.size() << endl;
            nonLeafNode->addNodes(vec);
            nonleafnodes->push_back(nonLeafNode);
        }
        // cout<< "here:" << i <<endl;
        if (leafnodes->size() > pageSize * nodeNum)
        {
            NonLeafNode *nonLeafNode = new NonLeafNode();
            nonLeafNode->level = i + 1;
            auto bn = leafnodes->begin() + pageSize * nodeNum;
            auto en = leafnodes->end();
            vector<nodespace::Node *> vec(bn, en);
            // cout << vec.size() << endl;
            nonLeafNode->addNodes(vec);
            nonleafnodes->push_back(nonLeafNode);
            expRecorder.non_leaf_node_num += 1;
        }
        leafnodes->clear();
        leafnodes->insert(leafnodes->begin(), nonleafnodes->begin(), nonleafnodes->end());
        if (i == level - 1)
        {
            root = (*nonleafnodes)[0];
        }
    }
    // cout << "level:" << root->level << endl;
    auto finish = chrono::high_resolution_clock::now();
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    // ExpRecorder expRecorder;
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    expRecorder.size = (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * (expRecorder.non_leaf_node_num + expRecorder.leaf_node_num) + expRecorder.dataset_cardinality * 2 * Constants::EACH_DIM_LENGTH * 2;
}

void HRR::point_query(ExpRecorder &expRecorder, Point queryPoint)
{
    queue<nodespace::Node *> nodes;
    nodes.push(root);
    bool isFound = false;
    while (!nodes.empty())
    {
        nodespace::Node *top = nodes.front();
        nodes.pop();

        if (typeid(*top) == typeid(NonLeafNode))
        {
            NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top);
            if (nonLeafNode->mbr.contains(queryPoint))
            {
                expRecorder.page_access += 1;
                // TODO opt
                for (nodespace::Node *node : *(nonLeafNode->children))
                {
                    nodes.push(node);
                }
            }
        }
        else if (typeid(*top) == typeid(LeafNode))
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
            if (leafNode->mbr.contains(queryPoint))
            {
                expRecorder.page_access += 1;
                // for (Point *point : *(leafNode->children))
                // {
                //     if (point == queryPoint)
                //     {
                //         // cout << "find it !" << endl;
                //         return;
                //     }
                // }
                vector<Point>::iterator iter = find(leafNode->children->begin(), leafNode->children->end(), queryPoint);
                if (iter != leafNode->children->end())
                {
                    return;
                }
            }
        }
        if (isFound)
        {
            break;
        }
    }
}

void HRR::point_query(ExpRecorder &expRecorder, vector<Point> queryPoints)
{
    cout << "HRR::point_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (long i = 0; i < queryPoints.size(); i++)
    {
        queue<nodespace::Node *> nodes;
        nodes.push(root);
        bool isFound = false;
        while (!nodes.empty())
        {
            nodespace::Node *top = nodes.front();
            nodes.pop();
            if (typeid(*top) == typeid(NonLeafNode))
            {
                NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top);
                if (nonLeafNode->mbr.contains(queryPoints[i]))
                {
                    expRecorder.page_access += 1;
                    // TODO opt
                    for (nodespace::Node *node : *(nonLeafNode->children))
                    {
                        nodes.push(node);
                    }
                }
            }
            else if (typeid(*top) == typeid(LeafNode))
            {
                LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
                if (leafNode->mbr.contains(queryPoints[i]))
                {
                    expRecorder.page_access += 1;
                    vector<Point>::iterator iter = find(leafNode->children->begin(), leafNode->children->end(), queryPoints[i]);
                    if (iter != leafNode->children->end())
                    {
                        break;
                    }
                }
            }
            if (isFound)
            {
                break;
            }
        }
    }
    auto finish = chrono::high_resolution_clock::now();
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / queryPoints.size();
    expRecorder.page_access = (double)expRecorder.page_access / queryPoints.size();
}

vector<Point> HRR::window_query(ExpRecorder &expRecorder, Mbr queryWindow)
{
    vector<Point> windowQueryResults;
    queue<nodespace::Node *> nodes;
    nodes.push(root);
    while (!nodes.empty())
    {
        nodespace::Node *top = nodes.front();
        nodes.pop();

        if (typeid(*top) == typeid(NonLeafNode))
        {
            NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top);
            if (nonLeafNode->mbr.interact(queryWindow))
            {
                expRecorder.page_access += 1;
                for (nodespace::Node *node : *(nonLeafNode->children))
                {
                    if (node->mbr.interact(queryWindow))
                    {
                        nodes.push(node);
                    }
                }
            }
        }
        else if (typeid(*top) == typeid(LeafNode))
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
            expRecorder.page_access += 1;
            for (Point point : *(leafNode->children))
            {
                if (queryWindow.contains(point))
                {
                    windowQueryResults.push_back(point);
                }
            }
        }
    }
    return windowQueryResults;
}

void HRR::window_query(ExpRecorder &expRecorder, vector<Mbr> queryWindows)
{
    int length = queryWindows.size();

    cout << "HRR::window_query length: " << length << endl;
    // length = 1;
    long long time = 0;
    int size = 0;
    for (int i = 0; i < length; i++)
    {
        // expRecorder.windowQueryResults.insert();
        auto start = chrono::high_resolution_clock::now();
        vector<Point> result = window_query(expRecorder, queryWindows[i]);
        auto finish = chrono::high_resolution_clock::now();
        time += chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
        size += result.size();
        // cout << "result.size(): " << result.size() << endl;
    }

    expRecorder.time = time / length;
    expRecorder.page_access = (double)expRecorder.page_access / length;
    cout << "time: " << expRecorder.time << endl;
    cout << "total size: " << size << endl;
}

vector<Point> HRR::kNN_query(ExpRecorder &expRecorder, Point queryPoint, int k)
{
    vector<Point> windowQueryResults;
    priority_queue<NodeExtend *, vector<NodeExtend *>, sortPQ> pq;
    priority_queue<NodeExtend *, vector<NodeExtend *>, sortPQ_Desc> pointPq;
    float maxP2PDist = numeric_limits<float>::max();
    NodeExtend *first = new NodeExtend(root, 0);
    pq.push(first);
    // float maxDist = numeric_limits<float>::max();

    while (!pq.empty())
    {
        NodeExtend *top = pq.top();
        pq.pop();
        if (top->is_leafnode())
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top->node);
            expRecorder.page_access += 1;
            for (Point point : *(leafNode->children))
            {
                float d = point.cal_dist(queryPoint);
                if (pointPq.size() >= k)
                {
                    if (d > maxP2PDist)
                    {
                        continue;
                    }
                    else
                    {
                        NodeExtend *temp = new NodeExtend(point, d);
                        pointPq.push(temp);
                        pointPq.pop();
                        maxP2PDist = pointPq.top()->dist;
                    }
                }
                else
                {
                    NodeExtend *temp = new NodeExtend(point, d);
                    pointPq.push(temp);
                }
            }
        }
        else
        {
            NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top->node);
            // queryPoint->print();
            expRecorder.page_access += 1;
            for (nodespace::Node *node : *(nonLeafNode->children))
            {
                float dist = node->cal_dist(queryPoint);
                if (dist > maxP2PDist)
                {
                    continue;
                }
                NodeExtend *temp = new NodeExtend(node, dist);
                pq.push(temp);
            }
        }
    }
    while (!pointPq.empty())
    {
        windowQueryResults.push_back(pointPq.top()->point);
        pointPq.pop();
    }
    return windowQueryResults;
}

void HRR::kNN_query(ExpRecorder &expRecorder, vector<Point> queryPoints, int k)
{
    cout << "HRR::kNN_query" << endl;
    auto start = chrono::high_resolution_clock::now();
    int length = queryPoints.size();
    for (int i = 0; i < length; i++)
    {
        priority_queue<Point, vector<Point>, sortForKNN2> temp_pq;
        expRecorder.pq = temp_pq;
        vector<Point> result = kNN_query(expRecorder, queryPoints[i], k);
        // for (size_t j = 0; j < result.size(); j++)
        // {
        //     cout<< result[j].getSelf() << " " << result[j].cal_dist(queryPoints[i]) << endl;
        // }
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / length;
    expRecorder.k_num = k;
    expRecorder.page_access = (double)expRecorder.page_access / length;
}

// change the implementation
void HRR::insert(ExpRecorder &expRecorder, Point point)
{
    // long x_i = SearchHelper::binarySearch<double>(xs, point->x);
    // long y_i = SearchHelper<double>::binarySearch(ys, point->y);
    // long long hVal = compute_Hilbert_value(x_i, y_i, side);

    // long index = SearchHelper<long long>::binarySearch(hs, hVal);
    nodespace::Node *head = root;
    queue<nodespace::Node *> nodes;
    nodes.push(root);
    while (!nodes.empty())
    {
        nodespace::Node *top = nodes.front();
        nodes.pop();
        if (typeid(*top) == typeid(NonLeafNode))
        {
            NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top);
            if (nonLeafNode->mbr.contains(point))
            {
                for (nodespace::Node *node : *(nonLeafNode->children))
                {
                    if (node->mbr.contains(point))
                    {
                        nodes.push(node);
                        break;
                    }
                }
            }
        }
        else if (typeid(*top) == typeid(LeafNode))
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
            if (leafNode->mbr.contains(point))
            {
                if (leafNode->is_full())
                {
                    nodespace::Node *rightNode = leafNode->split();
                    leafNode->add_point(point);

                    NonLeafNode *parent = leafNode->parent;
                    while (true)
                    {
                        if (parent->is_full())
                        {
                            NonLeafNode *rightParentNode = parent->split();
                            parent->addNode(rightNode);
                            if (parent == root)
                            {
                                NonLeafNode *newRoot = new NonLeafNode();
                                newRoot->addNode(parent);
                                newRoot->addNode(rightParentNode);
                                root = newRoot;
                                return;
                            }
                            else
                            {
                                parent = parent->parent;
                                rightNode = rightParentNode;
                            }
                        }
                        else
                        {
                            parent->addNode(rightNode);
                            return;
                        }
                    }
                }
                else
                {
                    leafNode->add_point(point);
                    return;
                }
            }
        }
    }
}

void HRR::insert(ExpRecorder &exp_recorder, vector<Point> points)
{
    // vector<Point> points = Point::get_inserted_points(exp_recorder.insert_num, exp_recorder.insert_points_distribution);
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        insert(exp_recorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long previous_time = exp_recorder.insert_time * exp_recorder.previous_insert_num;
    exp_recorder.previous_insert_num += points.size();
    exp_recorder.insert_time = (previous_time + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / exp_recorder.previous_insert_num;
}

void HRR::remove(ExpRecorder &expRecorder, Point point)
{
    nodespace::Node *head = root;
    queue<nodespace::Node *> nodes;
    nodes.push(root);
    while (!nodes.empty())
    {
        nodespace::Node *top = nodes.front();
        nodes.pop();
        if (typeid(*top) == typeid(NonLeafNode))
        {
            NonLeafNode *nonLeafNode = dynamic_cast<NonLeafNode *>(top);
            if (nonLeafNode->mbr.contains(point))
            {
                for (nodespace::Node *node : *(nonLeafNode->children))
                {
                    nodes.push(node);
                }
            }
        }
        else if (typeid(*top) == typeid(LeafNode))
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
            if (leafNode->mbr.contains(point) && leafNode->delete_point(point))
            {
                break;
            }
        }
    }
}

void HRR::remove(ExpRecorder &expRecorder, vector<Point> points)
{
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        remove(expRecorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    long long oldTimeCost = expRecorder.delete_time * expRecorder.delete_num;
    expRecorder.delete_num += points.size();
    expRecorder.delete_time = (oldTimeCost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / expRecorder.delete_num;
}