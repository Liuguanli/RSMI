#include <iostream>
#include <string.h>
#include "ExpRecorder.h"
#include "Constants.h"
using namespace std;

ExpRecorder::ExpRecorder()
{
}

string ExpRecorder::get_time()
{
    return "time:" + to_string(time) + "\n";
}

string ExpRecorder::get_time_size_errors()
{
    string result = "time:" + to_string(time) + "\n" + "size:" + to_string(size) + "\n" + "maxError:" + to_string(max_error) + "\n" + "min_error:" + to_string(min_error) + "\n" + "leaf_node_num:" + to_string(leaf_node_num) + "\n" + "average_max_error:" + to_string(average_max_error) + "\n" + "average_min_error:" + to_string(average_min_error) + "\n" + "depth:" + to_string(depth) + "\n";
    time = 0;
    size = 0;
    max_error = 0;
    min_error = 0;
    leaf_node_num = 0;
    depth = 0;
    return result;
}

string ExpRecorder::get_time_size()
{
    string result = "time:" + to_string(time) + "\n" + "size:" + to_string(size) + "\n";
    time = 0;
    size = 0;
    return result;
}

string ExpRecorder::get_time_accuracy()
{
    string result = "time:" + to_string(time) + "\n" + "accuracy:" + to_string(accuracy) + "\n";
    time = 0;
    accuracy = 0;
    return result;
}

string ExpRecorder::get_time_pageaccess_accuracy()
{
    string result = "time:" + to_string(time) + "\n" + "pageaccess:" + to_string(page_access) + "\n" + "accuracy:" + to_string(accuracy) + "\n";
    time = 0;
    page_access = 0;
    accuracy = 0;
    return result;
}

string ExpRecorder::get_time_pageaccess()
{
    string result = "time:" + to_string(time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_delete_time_pageaccess()
{
    string result = "time:" + to_string(delete_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_insert_time_pageaccess()
{
    string result = "time:" + to_string(insert_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_insert_time_pageaccess_rebuild()
{
    string result = "time:" + to_string(insert_time) + "\n" + "pageaccess:" + to_string(page_access) + "\n" + "rebuild_num:" + to_string(rebuild_num) + "\n" + "rebuild_time:" + to_string(rebuild_time) + "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_size()
{
    string result = "size:" + to_string(size) + "\n";
    size = 0;
    return result;
}

void ExpRecorder::cal_size()
{
    size = (Constants::DIM * Constants::PAGESIZE * Constants::EACH_DIM_LENGTH + Constants::PAGESIZE * Constants::INFO_LENGTH + Constants::DIM * Constants::DIM * Constants::EACH_DIM_LENGTH) * leaf_node_num + non_leaf_node_num * Constants::EACH_DIM_LENGTH;
}

void ExpRecorder::clean()
{
    index_high = 0;
    index_low = 0;

    leaf_node_num = 0;
    non_leaf_node_num = 0;

    window_query_result_size = 0;
    acc_window_query_qesult_size = 0;

    knn_query_results.clear();
    knn_query_results.shrink_to_fit();

    acc_knn_query_results.clear();
    acc_knn_query_results.shrink_to_fit();

    time = 0;
    page_access = 0;
    accuracy = 0;
    size = 0;

    window_query_results.clear();
    window_query_results.shrink_to_fit();

    rebuild_num = 0;
    rebuild_time = 0;
    max_error = 0;
    min_error = 0;

    average_min_error = 0;
    average_max_error = 0;

    last_level_model_num = 0;
    depth = 0;
}
