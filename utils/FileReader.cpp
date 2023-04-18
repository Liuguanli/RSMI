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

vector<vector<string>> FileReader::get_data()
{
    return get_data(this->filename);
}

vector<Point> FileReader::get_points()
{
    ifstream file(filename);
    vector<Point> points;
    string line = "";
    delimeter = "\t";

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
