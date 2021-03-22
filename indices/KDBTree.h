#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <queue>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include "../utils/ExpRecorder.h"
#include "../utils/SortTools.h"
#include "../entities/Mbr.h"
#include "../entities/Point.h"
#include "../entities/LeafNode.h"
#include "../entities/NonLeafNode.h"
#include "../entities/Node.h"
#include "../entities/NodeExtend.h"
#include "../utils/Constants.h"

class KDBTree
{
private:
    int level;
    NonLeafNode *root;
    int pageSize;
    int N;

public:
    KDBTree();
    KDBTree(int N);
    void build(ExpRecorder &expRecorder, vector<Point> points);
    void split(ExpRecorder &expRecorder, NonLeafNode *root, vector<Point> points, bool splitX, int level);

    void pointQuery(ExpRecorder &expRecorder, Point queryPoint);
    void pointQuery(ExpRecorder &expRecorder, vector<Point> queryPoints);

    void windowQuery(ExpRecorder &expRecorder, vector<Mbr> queryWindows);
    vector<Point> windowQuery(ExpRecorder &expRecorder, Mbr queryWindows);

    void kNNQuery(ExpRecorder &expRecorder, vector<Point> queryPoints, int k);
    vector<Point> kNNQuery(ExpRecorder &expRecorder, Point queryPoint, int k);

    void insert(ExpRecorder &expRecorder, Point);
    void insert(ExpRecorder &expRecorder, vector<Point>);

    void remove(ExpRecorder &expRecorder, Point);
    void remove(ExpRecorder &expRecorder, vector<Point>);
    ~KDBTree();
};

KDBTree::KDBTree()
{
    pageSize = Constants::PAGESIZE;
}

KDBTree::KDBTree(int N)
{
    pageSize = Constants::PAGESIZE;
    this->N = N;
}

void KDBTree::build(ExpRecorder &expRecorder, vector<Point> points)
{
    cout << "build: " << endl;
    auto start = chrono::high_resolution_clock::now();
    Mbr mbr(0.0, 0.0, 1.0, 1.0);
    root = new NonLeafNode(mbr);
    expRecorder.non_leaf_node_num++;
    root->level = 0;
    this->split(expRecorder, root, points, true, 1);
    auto finish = chrono::high_resolution_clock::now();
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count();
    expRecorder.cal_size();
    cout << "build finish: " << endl;
}

void KDBTree::split(ExpRecorder &expRecorder, NonLeafNode *root, vector<Point> points, bool splitX, int level)
{
    if (splitX)
    {
        sort(points.begin(), points.end(), sortX());
    }
    else
    {
        sort(points.begin(), points.end(), sortY());
    }
    int estimatedSize;
    int partitionNum;
    if (pageSize * pageSize < points.size())
    {
        estimatedSize = points.size() / pageSize;
        partitionNum = pageSize;
    }
    else
    {
        estimatedSize = pageSize;
        partitionNum = points.size() / pageSize;
        if (points.size() % pageSize != 0)
        {
            partitionNum++;
        }
    }
    // cout<< "level: " << level << " partitionNum: " << partitionNum << " estimatedSize: " << estimatedSize << endl;

    for (size_t i = 0; i < partitionNum; i++)
    {
        Mbr mbr = root->mbr;
        vector<Point> temp;
        if (i == partitionNum - 1)
        {
            auto bn = points.begin() + i * estimatedSize;
            auto en = points.end();
            vector<Point> vec(bn, en);
            temp = vec;
        }
        else
        {
            auto bn = points.begin() + i * estimatedSize;
            auto en = points.begin() + i * estimatedSize + estimatedSize;
            vector<Point> vec(bn, en);
            temp = vec;
        }
        Mbr tempMbr;
        if (splitX)
        {
            tempMbr.y1 = mbr.y1;
            tempMbr.y2 = mbr.y2;
            tempMbr.x1 = temp.front().x;
            tempMbr.x2 = temp.back().x;
        }
        else
        {
            tempMbr.x1 = mbr.x1;
            tempMbr.x2 = mbr.x2;
            tempMbr.y1 = temp.front().y;
            tempMbr.y2 = temp.back().y;
        }

        if (temp.size() > pageSize)
        {
            NonLeafNode *nonLeafNode = new NonLeafNode(tempMbr);
            expRecorder.non_leaf_node_num++;
            nonLeafNode->level = level;
            root->addNode(nonLeafNode);
            split(expRecorder, nonLeafNode, temp, !splitX, level + 1);
        }
        else
        {
            LeafNode *leafNode = new LeafNode(tempMbr);
            leafNode->add_points(temp);
            expRecorder.leaf_node_num++;
            leafNode->level = level;
            root->addNode(leafNode);
        }
    }
}

void KDBTree::pointQuery(ExpRecorder &expRecorder, Point queryPoint)
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
                for (Point point : *(leafNode->children))
                {
                    if (point == queryPoint)
                    {
                        // cout << "find it !" << endl;
                        return;
                    }
                }
            }
        }
        if (isFound)
        {
            break;
        }
    }
    cout<< "???" << endl;
}

void KDBTree::pointQuery(ExpRecorder &expRecorder, vector<Point> queryPoints)
{
    expRecorder.clean();
    // cout << "pointQuery:" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (Point point : queryPoints)
    {
        pointQuery(expRecorder, point);
    }
    auto finish = chrono::high_resolution_clock::now();
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / queryPoints.size();
    expRecorder.page_access = (double)expRecorder.page_access / queryPoints.size();
}

void KDBTree::windowQuery(ExpRecorder &expRecorder, vector<Mbr> queryWindows)
{
    // cout << "windowQuery:" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < queryWindows.size(); i++)
    {
        // expRecorder.window_query_results.insert();
        windowQuery(expRecorder, queryWindows[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / queryWindows.size();
    expRecorder.page_access = (double)expRecorder.page_access / queryWindows.size();
    cout<< "time: " << expRecorder.time << endl;
}

vector<Point> KDBTree::windowQuery(ExpRecorder &expRecorder, Mbr queryWindow)
{
    vector<Point> window_query_results;
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
                    nodes.push(node);
                }
            }
        }
        else if (typeid(*top) == typeid(LeafNode))
        {
            LeafNode *leafNode = dynamic_cast<LeafNode *>(top);
            if (leafNode->mbr.interact(queryWindow))
            {
                expRecorder.page_access += 1;
                for (Point point : *(leafNode->children))
                {
                    if (queryWindow.contains(point))
                    {
                        window_query_results.push_back(point);
                    }
                }
            }
        }
    }
    return window_query_results;
}

vector<Point> KDBTree::kNNQuery(ExpRecorder &expRecorder, Point queryPoint, int k)
{
    // vector<Point *> result;
    // float knnquerySide = sqrt((float)k / N);
    // while (true)
    // {
    //     Mbr *mbr = Mbr::getMbr(queryPoint, knnquerySide);
    //     vector<Point *> tempResult = windowQuery(expRecorder, mbr);
    //     if (tempResult.size() >= k)
    //     {
    //         sort(tempResult.begin(), tempResult.end(), sortForKNN(queryPoint));
    //         Point *last = tempResult[k - 1];
    //         if (last->cal_dist(queryPoint) <= knnquerySide)
    //         {
    //             // TODO get top K from the vector.
    //             auto bn = tempResult.begin();
    //             auto en = tempResult.begin() + k;
    //             vector<Point *> vec(bn, en);
    //             result = vec;
    //             break;
    //         }
    //     }
    //     knnquerySide = knnquerySide * 2;
    //     break;
    // }
    // return result;
    vector<Point> window_query_results;
    priority_queue<NodeExtend *, vector<NodeExtend *>, sortPQ> pq;
    priority_queue<NodeExtend *, vector<NodeExtend *>, sortPQ_Desc> pointPq;
    float maxP2PDist = numeric_limits<float>::max();

    NodeExtend *first = new NodeExtend(root, 0);
    pq.push(first);
    // double maxDist = numeric_limits<double>::max();

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
        window_query_results.push_back(pointPq.top()->point);
        pointPq.pop();
    }
    return window_query_results;
}

void KDBTree::kNNQuery(ExpRecorder &expRecorder, vector<Point> queryPoints, int k)
{
    // cout << "kNNQuery:" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < queryPoints.size(); i++)
    {
        kNNQuery(expRecorder, queryPoints[i], k);
    }
    auto finish = chrono::high_resolution_clock::now();
    // cout << "end:" << end.tv_nsec << " begin" << begin.tv_nsec << endl;
    expRecorder.time = chrono::duration_cast<chrono::nanoseconds>(finish - start).count()/ queryPoints.size();
    expRecorder.page_access = (double)expRecorder.page_access / queryPoints.size();
}

void KDBTree::insert(ExpRecorder &expRecorder, Point point)
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

void KDBTree::insert(ExpRecorder &expRecorder, vector<Point> points)
{
    // cout << "insert:" << endl;
    struct timespec begin, end, gap;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < points.size(); i++)
    {
        insert(expRecorder, points[i]);
    }
    auto finish = chrono::high_resolution_clock::now();
    long long oldTimeCost = expRecorder.insert_time * expRecorder.insert_num;
    expRecorder.insert_num += points.size();
    expRecorder.insert_time = (oldTimeCost + chrono::duration_cast<chrono::nanoseconds>(finish - start).count()) / expRecorder.insert_num;
}

void KDBTree::remove(ExpRecorder &expRecorder, Point point)
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

void KDBTree::remove(ExpRecorder &expRecorder, vector<Point> points)
{
    // cout << "remove:" << endl;
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

KDBTree::~KDBTree()
{
}
