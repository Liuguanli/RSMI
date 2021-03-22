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
    string result = "time:" + to_string(time) + "\n";
    result += "top_level_time:" + to_string(top_level_time) + "\n";
    result += "bottom_level_time:" + to_string(bottom_level_time) + "\n";
    result += "size:" + to_string(size) + "\n";
    result += "maxError:" + to_string(max_error) + "\n";
    result += "min_error:" + to_string(min_error) + "\n";
    result += "top_error:" + to_string(top_error) + "\n";
    result += "bottom_error:" + to_string(bottom_error) + "\n";
    result += "loss:" + to_string(loss) + "\n";
    result += "leaf_node_num:" + to_string(leaf_node_num) + "\n";
    result += "depth:" + to_string(depth) + "\n";
    // if (structure_name == "ZM" || structure_name == "RSMI")
    // {
    // }
    if (structure_name == "ZM-MR" || structure_name == "RSMI-MR")
    {
        result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
    }
    if (structure_name == "ZM-RL" || structure_name == "RSMI-RL")
    {
        result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
        result += "top_rl_time:" + to_string(top_rl_time) + "\n";
    }
    if (structure_name == "ZM-SPF" || structure_name == "ZM-SP" || structure_name == "RSMI-SP")
    {
        result += "sampling rate:" + to_string(sampling_rate) + "\n";
    }
    if (structure_name == "ZM-RS" || structure_name == "RSMI-RS")
    {
        result += "rep threshold m:" + to_string(representative_threshold_m) + "\n";
    }
    if (structure_name == "ZM-CL" || structure_name == "RSMI-CL")
    {
        result += "cluster_method:" + cluster_method + "\n";
        result += "cluster_size:" + to_string(cluster_size) + "\n";
    }
    result += "\n";
    time = 0;
    top_level_time = 0;
    bottom_level_time = 0;
    size = 0;
    max_error = 0;
    min_error = 0;
    top_error = 0;
    bottom_error = 0;
    leaf_node_num = 0;
    depth = 0;
    loss = 0;
    representative_threshold_m = 0;
    top_rl_time = 0;
    return result;
}

string ExpRecorder::get_time_size()
{
    string result = "time:" + to_string(time) + "\n" + "size:" + to_string(size) + "\n";
    time = 0;
    size = 0;
    result += "\n";
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
    string result = "time:" + to_string(time) + "\n";
    result += "prediction_time:" + to_string(prediction_time) + "\n";
    result += "sfc_cal_time:" + to_string(sfc_cal_time) + "\n";
    result += "search_time:" + to_string(search_time) + "\n";
    result += "search_steps:" + to_string(search_steps) + "\n";
    result += "point_not_found:" + to_string(point_not_found) + "\n";
    result += "pageaccess:" + to_string(page_access) + "\n";
    // if (structure_name == "ZM" || structure_name == "RSMI")
    // {
    // }
    if (structure_name == "ZM-MR" || structure_name == "RSMI-MR")
    {
        result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
    }
    if (structure_name == "ZM-RL" || structure_name == "RSMI-RL")
    {
        result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
    }
    if (structure_name == "ZM-SPF" || structure_name == "ZM-SP" || structure_name == "RSMI-SP")
    {
        result += "sampling rate:" + to_string(sampling_rate) + "\n";
    }
    if (structure_name == "ZM-RS" || structure_name == "RSMI-RS")
    {
        result += "rep threshold m:" + to_string(representative_threshold_m) + "\n";
    }
    if (structure_name == "ZM-CL" || structure_name == "RSMI-CL")
    {
        result += "cluster_method:" + cluster_method + "\n";
        result += "cluster_size:" + to_string(cluster_size) + "\n";
    }
    if (structure_name == "Grid" || structure_name == "HRR" || structure_name == "KDB")
    {
    }
    result += "\n";
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

    top_error = 0;
    bottom_error = 0;

    last_level_model_num = 0;
    depth = 0;

    bottom_level_time = 0;
    top_level_time = 0;
    search_steps = 0;
    search_time = 0;
}

string ExpRecorder::get_file_name()
{
    return distribution + "_" + to_string(dataset_cardinality) + "_" + to_string(skewness) + "_2_.csv";
}

string ExpRecorder::get_dataset_name()
{
    return Constants::DATASETS + get_file_name();
}

void ExpRecorder::set_structure_name(string prefix)
{
    string name = "";
    if (is_model_reuse)
    {
        name = "-MR";
    }
    else if (is_rl)
    {
        name = "-RL";
    }
    else if (is_sp)
    {
        if (is_sp_first)
        {
            name = "-SPF";
        }
        else
        {
            name = "-SP";
        }
    }
    else if (is_rs)
    {
        name = "-RS";
    }
    else if (is_cluster)
    {
        name = "-CL";
    }
    structure_name = prefix + name;
}

ExpRecorder* ExpRecorder::test_sp()
{
    test_reset();
    is_sp = true;
    return this;
}

ExpRecorder* ExpRecorder::test_sp_first()
{
    test_reset();
    is_sp = true;
    is_sp_first = true;
    return this;
}

ExpRecorder* ExpRecorder::test_model_reuse()
{
    test_reset();
    is_model_reuse = true;
    return this;
}

ExpRecorder* ExpRecorder::test_rl()
{
    test_reset();
    is_rl = true;
    return this;
}

ExpRecorder* ExpRecorder::test_cluster()
{
    test_reset();
    is_cluster = true;
    return this;
}

ExpRecorder* ExpRecorder::test_rs()
{
    test_reset();
    is_rs = true;
    return this;
}

void ExpRecorder::test_reset()
{
    is_sp = false;
    is_sp_first = false;
    is_model_reuse = false;
    is_rl = false;
    is_cluster = false;
    is_rs = false;
}
