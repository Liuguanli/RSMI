#include <string>
#include <iostream>
#include <fstream>

using namespace std;

namespace file_utils
{
    inline int check_dir(string path);
};

namespace file_utils
{
    int check_dir(string path)
    {
        std::ifstream fin(path);
        if (!fin)
        {
            string command = "mkdir -p " + path;
            return system(command.c_str());
        }
        return 0;
    }
}
