#include "FileReader.h"

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>

// #include "../entities/Point.cpp"
#include "../entities/Mbr.h"
using namespace std;

FileReader::FileReader()
{
}

FileReader::FileReader(string filename, string delimeter)
{
    this->filename = filename;
    this->delimeter = delimeter;
}

FileReader::FileReader(string delimeter)
{
    this->delimeter = delimeter;
}

vector<vector<string>> FileReader::get_data(string path)
{
    ifstream file(path);

    vector<vector<string>> data_list;

    string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        data_list.push_back(vec);
    }
    // Close the File
    file.close();

    return data_list;
}

void FileReader::get_rebuild_data(string path, vector<float> &parameters, vector<float> &labels)
{
    ifstream file(path);
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        int length = vec.size();
        for (size_t i = 0; i < length - 1; i++)
        {
            parameters.push_back(stod(vec[i]));
        }
        labels.push_back(stod(vec[length - 1]));
    }
    file.close();
}

void FileReader::get_cost_model_data(string path, vector<float> &parameters, vector<float> &build_time_labels, vector<float> &query_time_labels)
{
    ifstream file(path);
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        int length = vec.size();
        parameters.push_back(stod(vec[0]) / 6400);
        for (size_t i = 1; i < length - 2; i++)
        {
            // std::cout << "vec[i]: " << vec[i] << std::endl;
            parameters.push_back(stod(vec[i]));
        }
        // std::cout << "vec[length - 2]: " << vec[length - 2] << std::endl;
        build_time_labels.push_back(stod(vec[length - 2]));
        // std::cout << "vec[length - 1]: " << vec[length - 1] << std::endl;
        query_time_labels.push_back(stod(vec[length - 1]));
    }
    file.close();
}

vector<vector<string>> FileReader::get_data()
{
    return get_data(this->filename);
}

vector<Point> FileReader::get_points()
{
    ifstream file(filename);
    vector<Point> points;
    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        if (vec.size() > 1)
        {
            Point point(stod(vec[0]), stod(vec[1]));
            points.push_back(point);
        }
    }
    // Close the File
    file.close();

    return points;
}

vector<Mbr> FileReader::get_mbrs()
{
    ifstream file(filename);

    vector<Mbr> mbrs;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        Mbr mbr(stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3]));
        mbrs.push_back(mbr);
    }

    file.close();

    return mbrs;
}

vector<Point> FileReader::get_points(string filename, string delimeter)
{
    ifstream file(filename);

    vector<Point> points;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        Point point(stod(vec[0]), stod(vec[1]));
        points.push_back(point);
    }
    // Close the File
    file.close();

    return points;
}

vector<Mbr> FileReader::get_mbrs(string filename, string delimeter)
{
    ifstream file(filename);

    vector<Mbr> mbrs;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
        Mbr mbr(stod(vec[0]), stod(vec[1]), stod(vec[2]), stod(vec[3]));
        mbrs.push_back(mbr);
    }

    file.close();

    return mbrs;
}

vector<float> FileReader::read_features()
{
    return this->read_features(this->filename);
}

void FileReader::read_sfc(string filename, vector<int> &sfc, vector<float> &cdf)
{
    ifstream file(filename);

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        sfc.push_back(stoi(vec[0]));
        cdf.push_back(stof(vec[1]));
    }
    file.close();
}

void FileReader::read_sfc_2d(string filename, vector<float> &features, vector<float> &cdf)
{
    ifstream file(filename);

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        features.push_back(stoi(vec[0]));
        features.push_back(stoi(vec[1]));
        cdf.push_back(stof(vec[2]));
    }
    file.close();
}

vector<int> FileReader::read_sfc(string filename)
{
    ifstream file(filename);

    vector<int> sfc;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        sfc.push_back(stoi(vec[0]));
    }
    file.close();

    return sfc;
}

vector<float> FileReader::read_features(string filename)
{
    ifstream file(filename);

    vector<float> features;

    string line = "";
    while (getline(file, line))
    {
        vector<string> vec;
        boost::algorithm::split(vec, line, boost::is_any_of(","));
        features.push_back(stof(vec[0]));
    }
    file.close();

    return features;
}
