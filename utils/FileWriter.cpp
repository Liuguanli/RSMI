#include "FileWriter.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include "ExpRecorder.h"
#include "Constants.h"
#include "../entities/Point.h"
#include "../entities/Mbr.h"

using namespace std;

FileWriter::FileWriter(string filename)
{
    this->filename = filename;
}

void FileWriter::write_mbrs(vector<Mbr> mbrs, ExpRecorder expRecorder)
{
    ofstream write;
    // string folder = "window/";
    string folder = Constants::WINDOW;
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
    // string folder = "knn/";
    string folder = Constants::KNN;
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
    // string folder = "update/";
    string folder = Constants::UPDATE;
    write.open((filename + folder + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + ".csv"), ios::out);
    for (Point point : points)
    {
        write << point.get_self();
    }
    write.close();
}

void FileWriter::write_build(ExpRecorder expRecorder)
{
    // string folder = "build/";
    string folder = Constants::BUILD;
    ofstream write;
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
    // string folder = "point/";
    string folder = Constants::POINT;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_acc_window_query(ExpRecorder expRecorder)
{
    // string folder = "accwindow/";
    string folder = Constants::ACCWINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.window_size) + "_" + to_string(expRecorder.window_ratio) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_window_query(ExpRecorder expRecorder)
{
    // string folder = "window/";
    string folder = Constants::WINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.window_size) + "_" + to_string(expRecorder.window_ratio) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_acc_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "accknn/";
    string folder = Constants::ACCKNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.k_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "knn/";
    string folder = Constants::KNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.k_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert(ExpRecorder expRecorder)
{
    // string folder = "insert/";
    string folder = Constants::INSERT;
    ofstream write;
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
    // string folder = "delete/";
    string folder = Constants::DELETE;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_delete_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_point_query(ExpRecorder expRecorder)
{
    // string folder = "insertPoint/";
    string folder = Constants::INSERTPOINT;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_acc_window_query(ExpRecorder expRecorder)
{
    // string folder = "insertAccWindow/";
    string folder = Constants::INSERTACCWINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert_window_query(ExpRecorder expRecorder)
{
    // string folder = "insertWindow/";
    string folder = Constants::INSERTWINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_insert_acc_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "insertAccKnn/";
    string folder = Constants::INSERTACCKNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_insert_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "insertKnn/";
    string folder = Constants::INSERTKNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.insert_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_point_query(ExpRecorder expRecorder)
{
    // string folder = "delete_point/";
    string folder = Constants::DELETEPOINT;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess();
    write.close();
}

void FileWriter::write_delete_acc_window_query(ExpRecorder expRecorder)
{
    // string folder = "deleteAccWindow/";
    string folder = Constants::DELETEACCWINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_acc_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "deleteAccKnn/";
    string folder = Constants::DELETEACCKNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_window_query(ExpRecorder expRecorder)
{
    // string folder = "deleteWindow/";
    string folder = Constants::DELETEWINDOW;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}

void FileWriter::write_delete_kNN_query(ExpRecorder expRecorder)
{
    // string folder = "deleteKnn/";
    string folder = Constants::DELETEACCKNN;
    ofstream write;
    write.open((filename + folder + expRecorder.structure_name + "_" + expRecorder.distribution + "_" + to_string(expRecorder.dataset_cardinality) + "_" + to_string(expRecorder.skewness) + "_" + to_string(expRecorder.delete_num) + "_" + to_string(expRecorder.N) + ".txt"), ios::app);
    write << expRecorder.get_time_pageaccess_accuracy();
    write.close();
}