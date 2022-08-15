#include "test_download_data.h"

using namespace std;

int get(const string& url, const string& filename){

    string cmd = "pwd";
    
    const char* cmd_ptr = cmd.c_str();

    system(cmd_ptr);

    cmd = "cd ../tests/ && python3 download_data.py --url \"" + url + "\" --filename \"" + filename + "\"";
    
    cmd_ptr = cmd.c_str();
   
    return system(cmd_ptr);
} 