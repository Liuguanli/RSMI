#ifndef FILEREADER_H
#define FILEREADER_H
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
// #include <boost/algorithm/string.hpp>
using namespace std;
# include "../entities/Point.h"
# include "../entities/Mbr.h"
class FileReader
{
    string filename;
    string delimeter=",";

public:
    FileReader();
    FileReader(string, string);
    FileReader(string);
    vector<vector<string>> get_data(); 
    vector<vector<string>> get_data(string); 
    vector<Point> get_points();
    vector<Mbr> get_mbrs();
    vector<Point> get_points(string filename, string delimeter);
    vector<Mbr> get_mbrs(string filename, string delimeter);
    vector<float> read_features();
    vector<float> read_features(string filename);
    vector<int> read_sfc(string filename);
    void read_sfc(string filename, vector<int> &, vector<float> &);
    void read_sfc_2d(string filename, vector<float> &, vector<float> &);
    void get_cost_model_data(string, vector<float> &, vector<float> &, vector<float> &);
    void get_rebuild_data(string, vector<float> &, vector<float> &);
};

#endif