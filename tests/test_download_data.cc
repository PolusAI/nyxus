#include "test_download_data.h"

using namespace std;

int get(const string& url, const string& filename){

    string cmd = "cd .. && python3 tests/download_data.py --url \"" + url + "\" --filename \"" + filename + "\"";
    
    const char* cmd_ptr = cmd.c_str();
   
    system(cmd_ptr);

    return 0;
}