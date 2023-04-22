#ifndef INDEX_H
#define INDEX_H

#include <string>
#include "../utils/ExpRecorder.h"
#include "../entities/Point.h"

// namespace base_index
// {
class Index
{
public:
    string structure_name;
    // Index();
    // virtual ~Index() {}
    virtual void build(ExpRecorder &exp_recorder, vector<Point> points) {cout << "build Index" << endl;}
    virtual bool point_query(ExpRecorder &exp_recorder, Point query_point) { return true;}
    virtual void point_query(ExpRecorder &exp_recorder, vector<Point> query_points) {}

    virtual void window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows) {}
    virtual void window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, float boundary, int k, Point query_point, float &) {}
    virtual void window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window) {}
    virtual void kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k) {}
    vector<Point> kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
    {
        vector<Point> result;
        return result;
    }
};

// Index::Index()
// {
// }

// void Index::build(ExpRecorder &exp_recorder, vector<Point> points)
// {

// }

// bool Index::point_query(ExpRecorder &exp_recorder, Point query_point)
// {
// }

// void Index::point_query(ExpRecorder &exp_recorder, vector<Point> query_points)
// {
// }

// void Index::window_query(ExpRecorder &exp_recorder, vector<Mbr> query_windows)
// {
// }

// void Index::window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window, float boundary, int k, Point query_point, float &)
// {
// }

// void Index::window_query(ExpRecorder &exp_recorder, vector<Point> vertexes, Mbr query_window)
// {
// }

// void Index::kNN_query(ExpRecorder &exp_recorder, vector<Point> query_points, int k)
// {
// }

// vector<Point> Index::kNN_query(ExpRecorder &exp_recorder, Point query_point, int k)
// {
//     vector<Point> result;
//     return result;
// }

// };

#endif