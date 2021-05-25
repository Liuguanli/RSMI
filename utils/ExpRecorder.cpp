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
    result += "level:" + to_string(level) + "\n";
    result += "top_level_time:" + to_string(top_level_time) + "\n";
    result += "bottom_level_time:" + to_string(bottom_level_time) + "\n";
    result += "sfc_cal_time:" + to_string(sfc_cal_time) + "\n";
    result += "ordering_cost:" + to_string(ordering_cost) + "\n";
    result += "training_cost:" + to_string(training_cost) + "\n";
    result += "extra_time:" + to_string(extra_time) + "\n";
    result += "cost_model_time:" + to_string(cost_model_time) + "\n";
    result += "upper_level_lambda:" + to_string(upper_level_lambda) + "\n";
    result += "lower_level_lambda:" + to_string(lower_level_lambda) + "\n";
    result += "cluster_size:" + to_string(cluster_size) + "\n";
    result += "sampling_rate:" + to_string(sampling_rate) + "\n";
    result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
    result += "top_rl_time:" + to_string(top_rl_time) + "\n";
    result += "rep threshold m:" + to_string(rs_threshold_m) + "\n";
    result += "cluster_method:" + cluster_method + "\n";
    result += "representative_threshold_m:" + to_string(rs_threshold_m) + "\n";
    result += "model_reuse_threshold:" + to_string(model_reuse_threshold) + "\n";
    result += "size:" + to_string(size) + "\n";
    result += "maxError:" + to_string(max_error) + "\n";
    result += "min_error:" + to_string(min_error) + "\n";
    result += "top_error:" + to_string(top_error) + "\n";
    result += "bottom_error:" + to_string(bottom_error) + "\n";
    result += "loss:" + to_string(loss) + "\n";
    result += "leaf_node_num:" + to_string(leaf_node_num) + "\n";
    result += "last_level_model_num:" + to_string(last_level_model_num) + "\n";
    result += "non_leaf_model_num:" + to_string(non_leaf_model_num) + "\n";
    result += "sp_num:" + to_string(sp_num) + "\n";
    result += "model_reuse_num:" + to_string(model_reuse_num) + "\n";
    result += "rl_num:" + to_string(rl_num) + "\n";
    result += "cluster_num:" + to_string(cluster_num) + "\n";
    result += "rs_num:" + to_string(rs_num) + "\n";
    result += "original:" + to_string(original_num) + "\n";
    result += "depth:" + to_string(depth) + "\n";
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
    rs_threshold_m = 0;
    top_rl_time = 0;
    extra_time = 0;
    ordering_cost = 0;
    sfc_cal_time = 0;
    training_cost = 0;
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
    result += "insert_num:" + to_string(previous_insert_num) + "\n";
    if (insert_num != 0)
    {
        result += "insert_num:" + to_string(previous_insert_num) + "\n";
        result += "insert_ratio:" + to_string(previous_insert_num * 100.0 / dataset_cardinality) + "%\n";
        result += "insert_points_distribution:" + insert_points_distribution + "\n";
    }
    result += "level:" + to_string(level) + "\n";
    result += "insert_points_distribution:" + insert_points_distribution + "\n";
    result += "upper_level_lambda:" + to_string(upper_level_lambda) + "\n";
    result += "lower_level_lambda:" + to_string(lower_level_lambda) + "\n";
    result += "cluster_size:" + to_string(cluster_size) + "\n";
    result += "sampling_rate:" + to_string(sampling_rate) + "\n";
    result += "representative_threshold_m:" + to_string(rs_threshold_m) + "\n";
    result += "rep threshold m:" + to_string(rs_threshold_m) + "\n";
    result += "cluster_method:" + cluster_method + "\n";
    result += "model_reuse_threshold:" + to_string(model_reuse_threshold) + "\n";
    result += "prediction_time:" + to_string(prediction_time) + "\n";
    result += "epsilon:" + to_string(model_reuse_threshold) + "\n";
    result += "sfc_cal_time:" + to_string(sfc_cal_time) + "\n";
    result += "search_time:" + to_string(search_time) + "\n";
    result += "search_steps:" + to_string(search_steps) + "\n";
    result += "point_not_found:" + to_string(point_not_found) + "\n";
    result += "pageaccess:" + to_string(page_access) + "\n";
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
    string result = "time:" + to_string(insert_time) + "\n";
    if (insert_num != 0)
    {
        result += "insert_num:" + to_string(previous_insert_num) + "\n";
        result += "insert_ratio:" + to_string(previous_insert_num * 100.0 / dataset_cardinality) + "%\n";
        result += "insert_points_distribution:" + insert_points_distribution + "\n";
    }
    result += "level:" + to_string(level) + "\n";
    result += "pageaccess:" + to_string(page_access) + "\n";
    result += "cluster_size:" + to_string(cluster_size) + "\n";
    result += "sampling_rate:" + to_string(sampling_rate) + "\n";
    result += "upper_level_lambda:" + to_string(upper_level_lambda) + "\n";
    result += "lower_level_lambda:" + to_string(lower_level_lambda) + "\n";
    result += "representative_threshold_m:" + to_string(rs_threshold_m) + "\n";
    result += "model_reuse_threshold:" + to_string(model_reuse_threshold) + "\n";
    result += "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_insert_time_pageaccess_rebuild()
{
    string result = "time:" + to_string(insert_time) + "\n";
    if (insert_num != 0)
    {
        result += "insert_num:" + to_string(previous_insert_num) + "\n";
        result += "insert_ratio:" + to_string(previous_insert_num * 100.0 / dataset_cardinality) + "%\n";
        result += "insert_points_distribution:" + insert_points_distribution + "\n";
    }
    result += "prediction_time:" + to_string(prediction_time) + "\n";
    result += "level:" + to_string(level) + "\n";
    result += "pageaccess:" + to_string(page_access) + "\n";
    result += "rebuild_num:" + to_string(rebuild_num) + "\n";
    result += "rebuild_time:" + to_string(rebuild_time) + "\n";
    result += "cluster_size:" + to_string(cluster_size) + "\n";
    result += "sampling_rate:" + to_string(sampling_rate) + "\n";
    result += "upper_level_lambda:" + to_string(upper_level_lambda) + "\n";
    result += "lower_level_lambda:" + to_string(lower_level_lambda) + "\n";
    result += "representative_threshold_m:" + to_string(rs_threshold_m) + "\n";
    result += "model_reuse_threshold:" + to_string(model_reuse_threshold) + "\n";
    result += "\n";
    time = 0;
    page_access = 0;
    return result;
}

string ExpRecorder::get_size()
{
    string result = "size:" + to_string(size) + "\n";
    result += "\n";
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
    extra_time = 0;
    point_not_found = 0;

    sfc_cal_time = 0;
    cost_model_time = 0;

    insert_time = 0;
    previous_insert_num = 0;

    sp_num = 0;
    model_reuse_num = 0;
    rl_num = 0;
    cluster_num = 0;
    rs_num = 0;
    original_num = 0;
    traverse_time = 0;
    test_reset();
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
    if (is_sp)
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
    else if (is_rl)
    {
        name = "-RL";
    }
    else if (is_model_reuse)
    {
        name = "-MR";
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

ExpRecorder *ExpRecorder::test_sp()
{
    test_reset();
    is_sp = true;
    return this;
}

ExpRecorder *ExpRecorder::test_sp_mr()
{
    test_reset();
    is_sp = true;
    is_model_reuse = true;
    return this;
}

ExpRecorder *ExpRecorder::test_sp_first()
{
    test_reset();
    is_sp = true;
    is_sp_first = true;
    return this;
}

ExpRecorder *ExpRecorder::test_model_reuse()
{
    test_reset();
    is_model_reuse = true;
    return this;
}

ExpRecorder *ExpRecorder::test_rl()
{
    test_reset();
    is_rl = true;
    return this;
}

ExpRecorder *ExpRecorder::test_rl_mr()
{
    test_reset();
    is_model_reuse = true;
    is_rl = true;
    return this;
}

ExpRecorder *ExpRecorder::test_cluster()
{
    test_reset();
    is_cluster = true;
    return this;
}

ExpRecorder *ExpRecorder::test_rs_mr()
{
    test_reset();
    is_rs = true;
    is_model_reuse = true;
    return this;
}

ExpRecorder *ExpRecorder::test_rs()
{
    test_reset();
    is_rs = true;
    return this;
}

ExpRecorder *ExpRecorder::set_level(int level)
{
    this->level = level;
    return this;
}

ExpRecorder *ExpRecorder::set_cost_model(bool is_cost_model)
{
    this->is_cost_model = is_cost_model;
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
