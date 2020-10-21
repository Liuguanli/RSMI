#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include "ExpRecorder.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"
// #include <boost/algorithm/Mbr.hpp>
using namespace std;
class FileWriter
{
    string filename;

public:
    FileWriter(string);
    void write_build(ExpRecorder);
    void write_point_query(ExpRecorder);
    void write_window_query(ExpRecorder);
    void write_kNN_query(ExpRecorder);
    void write_acc_window_query(ExpRecorder);
    void write_acc_kNN_query(ExpRecorder);
    void write_insert(ExpRecorder);
    void write_delete(ExpRecorder);

    void write_insert_point_query(ExpRecorder);
    void write_insert_window_query(ExpRecorder);
    void write_insert_acc_window_query(ExpRecorder);
    void write_insert_kNN_query(ExpRecorder);
    void write_insert_acc_kNN_query(ExpRecorder);

    void write_delete_point_query(ExpRecorder);
    void write_delete_window_query(ExpRecorder);
    void write_delete_kNN_query(ExpRecorder);
    void write_delete_acc_window_query(ExpRecorder);
    void write_delete_acc_kNN_query(ExpRecorder);

    void write_mbrs(vector<Mbr> mbrs, ExpRecorder expRecorder);
    void write_points(vector<Point> points, ExpRecorder expRecorder);
    void write_inserted_points(vector<Point> points, ExpRecorder expRecorder);
};
