#ifndef SHARD_CPP
#define SHARD_CPP

#include "LeafNode.h"
#include <vector>

class Shard
{
private:
    int id;
    int page_size;
    vector<LeafNode> pages;

    vector<int> PA;
    vector<float> PM;

public:
    Shard(int);
    Shard(int, int);
    void create_pages();
    void gen_local_model(vector<Point> points, int &id);
    bool search_point(ExpRecorder &exp_recorder, Point query_point);
    vector<Point> window_query(ExpRecorder &exp_recorder, Mbr query_window);
    void insert(ExpRecorder &exp_recorder, Point);

    int traverse();
};

Shard::Shard(int page_size)
{
    this->page_size = page_size;
}

Shard::Shard(int id, int page_size)
{
    this->page_size = page_size;
    this->id = id;
}

bool Shard::search_point(ExpRecorder &exp_recorder, Point query_point)
{
    // for (int i = 0; i < page_num; i++)
    // {
    //     cout << "page: " << i << endl;
    //     for (size_t j = 0; j < pages[i].children->size(); j++)
    //     {
    //         cout << "page: " << i << " item: " << j << endl;
    //         (*pages[i].children)[j].print();
    //     }
    // }
    int page_num = pages.size();
    if (PM.size() == 0)
    {
        vector<Point>::iterator iter = find(pages[0].children->begin(), pages[0].children->end(), query_point);
        exp_recorder.page_access++;
        if (iter != pages[0].children->end())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    for (int i = 0; i < page_num - 1; i++)
    {
        if (query_point.key < PM[i])
        {
            exp_recorder.page_access++;

            vector<Point>::iterator iter = find(pages[i].children->begin(), pages[i].children->end(), query_point);
            if (iter != pages[i].children->end())
            {
                return true;
            }
        }
        else if (query_point.key == PM[i])
        {
            exp_recorder.page_access++;

            vector<Point>::iterator iter = find(pages[i].children->begin(), pages[i].children->end(), query_point);
            if (iter != pages[i].children->end())
            {
                return true;
            }

            exp_recorder.page_access++;

            vector<Point>::iterator iter1 = find(pages[i + 1].children->begin(), pages[i + 1].children->end(), query_point);
            if (iter1 != pages[i + 1].children->end())
            {
                return true;
            }
        }
    }
    exp_recorder.page_access++;

    vector<Point>::iterator iter = find(pages[page_num - 1].children->begin(), pages[page_num - 1].children->end(), query_point);
    if (iter != pages[page_num - 1].children->end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

vector<Point> Shard::window_query(ExpRecorder &exp_recorder, Mbr query_window)
{
    int page_num = pages.size();
    vector<Point> res;
    for (size_t i = 0; i < page_num; i++)
    {
        exp_recorder.page_access++;
        for (Point point : *pages[i].children)
        {
            if (query_window.contains(point))
            {
                // res.push_back(point);
                exp_recorder.window_query_results.push_back(point);
            }
            // if (point.y >= query_window.y1 && point.y <= query_window.y2)
            // {
            //     // res.push_back(point);
            //     exp_recorder.window_query_results.push_back(point);
            // }
        }
    }
    return res;
}

void Shard::gen_local_model(vector<Point> points, int &id)
{
    int size = points.size();
    int page_num = ceil(size * 1.0 / page_size);
    // std::cout << " page_num: " << page_num << std::endl;
    for (size_t i = 0; i < page_num; i++)
    {
        PA.push_back(id);
        LeafNode leafNode(id++);
        int point_begin = i * page_size;
        int point_end = (i == (page_num - 1)) ? size : (i + 1) * page_size;
        vector<Point> shard_page_points(points.begin() + point_begin, points.begin() + point_end);
        leafNode.add_points(shard_page_points);
        pages.push_back(leafNode);
        if (i > 0)
        {
            PM.push_back(points[point_begin].key);
        }
    }
}

void Shard::insert(ExpRecorder &exp_recorder, Point point)
{
    int page_num = pages.size();
    if (PM.size() == 0)
    {
        for (size_t i = 0; i < pages[0].children->size(); i++)
        {
            if ((*pages[0].children)[i].key > point.key || ((*pages[0].children)[i].key == point.key && (*pages[0].children)[i].x >= point.x))
            {
                pages[0].children->insert(pages[0].children->begin() + i, point);
                break;
            }
            if (i == (pages[0].children->size() - 1))
            {
                pages[0].children->insert(pages[0].children->end(), point);
            }
        }
        if (pages[0].children->size() > page_size)
        {
            sort(pages[0].children->begin(), pages[0].children->end(), sort_key());
            LeafNode right = pages[0].split1();
            pages.push_back(right);
            PM.push_back((*right.children)[0].key);
        }
    }
    else
    {
        bool is_inserted = false;
        for (int i = 0; i < page_num - 1; i++)
        {
            if (point.key <= PM[i])
            {
                for (size_t j = 0; j < pages[i].children->size(); j++)
                {
                    if ((*pages[i].children)[j].key > point.key || ((*pages[i].children)[j].key == point.key && (*pages[i].children)[j].x >= point.x))
                    {
                        is_inserted = true;
                        pages[i].children->insert(pages[i].children->begin() + j, point);
                        break;
                    }
                    if (j == (pages[0].children->size() - 1))
                    {
                        if ((*pages[i + 1].children)[0].key >= point.key)
                        {
                            continue;
                        }
                        else
                        {
                            is_inserted = true;
                            pages[i].children->insert(pages[i].children->end(), point);
                            break;
                        }
                    }
                }
                if (pages[i].children->size() > page_size)
                {
                    sort(pages[i].children->begin(), pages[i].children->end(), sort_key());
                    LeafNode right = pages[i].split1();
                    pages.insert(pages.begin() + i + 1, right);
                    PM.insert(PM.begin() + i, (*right.children)[0].key);
                }
            }
            if (is_inserted)
            {
                return;
            }
        }
        for (size_t i = 0; i < pages[page_num - 1].children->size(); i++)
        {
            if ((*pages[page_num - 1].children)[i].key > point.key || ((*pages[page_num - 1].children)[i].key == point.key && (*pages[page_num - 1].children)[i].x >= point.x))
            {
                is_inserted = true;
                pages[page_num - 1].children->insert(pages[page_num - 1].children->begin() + i, point);
                break;
            }
            if (i == pages[page_num - 1].children->size() - 1)
            {
                is_inserted = true;
                pages[page_num - 1].children->insert(pages[page_num - 1].children->end(), point);
                break;
            }
        }
        if (pages[page_num - 1].children->size() > page_size)
        {
            LeafNode right = pages[page_num - 1].split1();
            pages.push_back(right);
            PM.push_back((*right.children)[0].key);
        }
        if (!is_inserted)
        {
            cout << "not inserted successfully" << endl;
        }
    }
}

#endif