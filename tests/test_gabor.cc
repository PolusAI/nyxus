#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "test_gabor.h"
#include "test_download_data.h"

namespace fs = std::filesystem;

using namespace std;

void assert_gpu_results(const std::string cpu_dir, const std::string gpu_dir){
    ifstream gpu_in(gpu_dir + "/NyxusFeatures.csv");
    ifstream cpu_in(cpu_dir + "/NyxusFeatures.csv");

    std::string gpu, cpu;

    int count = 0;
    while(true) {
        getline(gpu_in, gpu);
        getline(cpu_in, cpu);
        
        if(gpu == "" || cpu == "") break;

        if(gpu != cpu){
            cout << "gpu: " << gpu << endl;
            cout << "cpu: " << cpu << endl;
            continue;
        }

        ASSERT_EQ(gpu, cpu);
    }
}

void test_gabor_gpu_2018(){

    if (!fs::exists("dsb2018")) {
        if (!fs::exists("dsb2018.zip")) {
            get("https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip", "dsb2018.zip");
        } else {
            system("unzip dsb2018.zip");
        }
    } else {
        cout << "found directory" << endl;
    }

    std::string cpu_out = "../cpu_out";
    std::string gpu_out = "../gpu_out";

    fs::create_directory(gpu_out);
    fs::create_directory(cpu_out);

    std::string args = "./nyxus --verbosity=0 --features=GABOR --intDir=tests/dsb2018/train/images --segDir=tests/dsb2018/train/masks --outDir=gpu_out --filePattern=.* --csvFile=singlecsv --loaderThreads=1 --reduceThreads=1 --useGpu=true";

    args = "cd .. && " + args;
    const char* cmd = args.c_str();

    system(cmd);

    args = "./nyxus --verbosity=0 --features=GABOR --intDir=tests/dsb2018/train/images --segDir=tests/dsb2018/train/masks --outDir=cpu_out --filePattern=.* --csvFile=singlecsv --loaderThreads=1 --reduceThreads=8 --useGpu=false";
    args = "cd .. && " + args;

    cmd = args.c_str();

    system(cmd);

    assert_gpu_results(cpu_out, gpu_out);
}
