#include "FileWriter.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include "ExpRecorder.h"
#include "ExpRecorder.h"
#include "util.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"

using namespace std;

FileWriter::FileWriter(string filename)
{
    this->filename = filename;
    file_utils::check_dir(filename);
}

void FileWriter::write_counted_SFC(vector<int> values, string name)
{
    ofstream write;
    file_utils::check_dir(filename);
    write.open((filename + name), ios::out);
    int N = values.size();
    for (size_t i = 0; i < N; i++)
    {
        write << values[i] << endl;
    }
    write.close();
}

void FileWriter::write_Approximate_SFC(vector<float> z_values, vector<float> cdf, string name)
{
    ofstream write;
    file_utils::check_dir(filename);
    write.open((filename + name), ios::out);
    int N = z_values.size();
    for (size_t i = 0; i < N; i++)
    {
        write << z_values[i] << "," << cdf[i] << endl;
    }
    write.close();
}

void FileWriter::write_weighted_SFC(vector<float> values, string name)
{
    ofstream write;
    file_utils::check_dir(filename);
    write.open((filename + name), ios::out);
    int N = values.size();
    double sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        sum += (double)values[i];
        write << values[i] << "," << sum << endl;
    }
    write.close();
}

void FileWriter::write_SFC(vector<float> values, string name)
{
    ofstream write;
    file_utils::check_dir(filename);
    write.open((filename + name), ios::out);
    int N = values.size();
    if (N > 1000000)
    {
        for (size_t i = 0; i < N; i++)
        {
            // if ((i + 1) % 100 == 0)
            // {
                write << values[i] << "," << (i + 1.0) / N << endl;
            // }
        }
    }
    else
    {
        for (size_t i = 0; i < N; i++)
        {
            write << values[i] << "," << (i + 1.0) / N << endl;
        }
    }
    write.close();
}

void FileWriter::write_mbrs(vector<Mbr> mbrs, ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::WINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.window_size) + "_" + to_string(expRecorder.window_ratio) + ".csv"), ios::out);
    for (Mbr mbr : mbrs)
    {
        write << mbr.get_self();
    }
    write.close();
}
void FileWriter::write_points(vector<Point> points, ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::KNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + ".csv"), ios::out);
    for (Point point : points)
    {
        write << point.get_self();
    }
    write.close();
}
void FileWriter::write_inserted_points(vector<Point> points, ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::UPDATE;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + ".csv"), ios::out);
    for (Point point : points)
    {
        write << point.get_self();
    }
    write.close();
}

void FileWriter::write_build(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::BUILD;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    if (expRecorder.structure_name == "ZM" || expRecorder.structure_name == "RSMI")
    {
        write << expRecorder.get_time_size_errors();
    }
    else
    {
        write << expRecorder.get_time_size();
    }
    write.close();
}

void FileWriter::write_point_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::POINT;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_acc_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::ACCWINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.window_size) + "_" + to_string(expRecorder.window_ratio) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::WINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.window_size) + "_" + to_string(expRecorder.window_ratio) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_acc_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::ACCKNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.k_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::KNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.k_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERT;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    if (expRecorder.structure_name == "RSMI")
    {
        write << expRecorder.get_insert_time_pageaccess_rebuild();
    }
    else
    {
        write << expRecorder.get_insert_time_pageaccess();
    }
    write.close();
}

void FileWriter::write_delete(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETE;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_delete_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_point_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERTPOINT;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_acc_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERTACCWINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERTWINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert_acc_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERTACCKNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::INSERTKNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_point_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETEPOINT;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_delete_acc_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETEACCWINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_acc_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETEACCKNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_window_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETEWINDOW;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_kNN_query(ExpRecorder expRecorder)
{
    ofstream write;
    string folder = Constants::DELETEACCKNN;
    file_utils::check_dir(filename + folder);
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}