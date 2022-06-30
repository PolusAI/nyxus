#include "test_download_data.h"
#include <curl/curl.h>

using namespace std;

int get(const string& url, const string& filename){
    string cmd = "wget " + url;
    
    const char* cmd_ptr = cmd.c_str();
    system(cmd_ptr);

    cmd = "unzip " + filename;
    cmd_ptr = cmd.c_str();

    system(cmd_ptr);

    return 0;
}