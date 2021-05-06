#include <vector>
#include "LeafNode.h"
#include "Mbr.h"
#include "../utils/SortTools.h"

class Bucket
{

private:
    // void addNode(LeafNode *);
    void addNode(LeafNode);

public:
    Mbr mbr;
    // vector<LeafNode *> LeafNodes;
    vector<LeafNode> LeafNodes;
    Bucket();
    Bucket(Mbr mbr);
    bool point_query(ExpRecorder &expRecorder, Point);
    void windowQuery(ExpRecorder &expRecorder, Mbr);
    bool insert(Point);
    void remove(Point);
    vector<Point> getAllPoints();
    void split();
    bool splitX = true;

    vector<Point> dynArray;
};

Bucket::Bucket(Mbr mbr)
{
    this->mbr = mbr;
    LeafNode LeafNode1(mbr);
    addNode(LeafNode1);
}

Bucket::Bucket()
{
    LeafNode LeafNode;
    addNode(LeafNode);
}

bool Bucket::point_query(ExpRecorder &expRecorder, Point point)
{
    // vector<Point>::iterator iter = find(dynArray.begin(), dynArray.end(), point);
    // if (iter != dynArray.end())
    // {
    //     return true;
    // }

    for (LeafNode LeafNode : LeafNodes)
    {
        expRecorder.page_access++;
        vector<Point>::iterator iter = find(LeafNode.children->begin(), LeafNode.children->end(), point);
        if (iter != LeafNode.children->end())
        {
            return true;
        }
    }
    return false;
}

void Bucket::windowQuery(ExpRecorder &expRecorder, Mbr queryWindow)
{
    // vector<Point> window_query_results;
    // for (size_t i = 0; i < LeafNodes.size(); i++)
    // {
    //     if (LeafNodes[i].mbr.interact(queryWindow))
    //     {
    //         expRecorder.page_access++;
    //         for (Point point : *(LeafNodes[i].children))
    //         {
    //             if (queryWindow.contains(point))
    //             {
    //                 expRecorder.window_query_results.push_back(point);
    //                 // window_query_results.push_back(point);
    //             }
    //         }
    //     }
    // }

    for (Point point : dynArray)
    {
        if (queryWindow.contains(point))
        {
            expRecorder.window_query_results.push_back(point);
            // window_query_results.push_back(point);
        }
    }
    

    // for (LeafNode LeafNode : LeafNodes)
    // {
    //     if (LeafNode.mbr.interact(queryWindow))
    //     {
    //         expRecorder.page_access++;
    //         for (Point point : *(LeafNode.children))
    //         {
    //             if (queryWindow.contains(point))
    //             {
    //                 expRecorder.window_query_results.push_back(point);
    //                 // window_query_results.push_back(point);
    //             }
    //         }
    //     }
    // }
    // return window_query_results;
}

bool Bucket::insert(Point point)
{

    dynArray.push_back(point);

    bool isSplit = false;
    bool isAdded = false;
    for (size_t i = 0; i < LeafNodes.size(); i++)
    {
        if (LeafNodes[i].mbr.contains(point))
        {
            LeafNodes[i].children->push_back(point);
            isAdded = true;
            if (LeafNodes[i].children->size() > Constants::PAGESIZE)
            {
                Mbr oldMbr = LeafNodes[i].mbr;
                // new LeafNode
                LeafNode newSplit;
                int mid = 0;
                if (splitX)
                {
                    sort(LeafNodes[i].children->begin(), LeafNodes[i].children->end(), sortX());
                    float midX = (oldMbr.x1 + oldMbr.x2) / 2;
                    Mbr mbr(midX, oldMbr.y1, oldMbr.x2, oldMbr.y2);
                    oldMbr.x2 = midX;
                    newSplit.mbr = mbr;
                    isSplit = true;
                    for (size_t j = 0; j < LeafNodes[i].children->size(); j++)
                    {
                        if ((*(LeafNodes[i].children))[j].x > midX)
                        {
                            mid = j;
                            break;
                        }
                    }
                }
                else
                {
                    sort(LeafNodes[i].children->begin(), LeafNodes[i].children->end(), sortY());
                    float midY = (oldMbr.y1 + oldMbr.y2) / 2;
                    Mbr mbr(oldMbr.x1, midY, oldMbr.x2, oldMbr.y2);
                    oldMbr.y2 = midY;
                    newSplit.mbr = mbr;
                    isSplit = true;
                    for (size_t j = 0; j < LeafNodes[i].children->size(); j++)
                    {
                        if ((*(LeafNodes[i].children))[j].y > midY)
                        {
                            mid = j;
                            break;
                        }
                    }
                }

                vector<Point> vec(LeafNodes[i].children->begin() + mid, LeafNodes[i].children->end());
                newSplit.children->insert(newSplit.children->end(), vec.begin(), vec.end());
                // cout<< "newSplit->children size: " << newSplit->children->size() << endl; 
                // old LeafNode
                vector<Point> vec1(LeafNodes[i].children->begin(), LeafNodes[i].children->begin() + mid);
                LeafNodes[i].children->clear();
                LeafNodes[i].children->insert(LeafNodes[i].children->end(), vec1.begin(), vec1.end());
                // cout<< "LeafNodes[i]->children size: " << LeafNodes[i]->children->size() << endl;
                LeafNodes.insert(LeafNodes.begin() + i, newSplit);

                splitX = !splitX;
            }
            break;
        }
    }
    
    return isSplit;
}

vector<Point> Bucket::getAllPoints()
{
    vector<Point> result;
    for (LeafNode LeafNode : LeafNodes)
    {
        vector<Point> *tempResult = LeafNode.children;
        result.insert(result.end(), tempResult->begin(), tempResult->end());
    }
    return result;
}

void Bucket::remove(Point point)
{
    for (LeafNode LeafNode : LeafNodes)
    {
        if (LeafNode.mbr.contains(point) && LeafNode.delete_point(point))
        {
            // cout << "remove it" << endl;
            break;
        }
    }
}

// void Bucket::addNode(LeafNode *node)
// {
//     // add
//     LeafNodes.push_back(node);
//     // update MBR
//     mbr.update(node->mbr);
// }

void Bucket::addNode(LeafNode node)
{
    // add
    LeafNodes.push_back(node);
    // update MBR
    mbr.update(node.mbr);
}
