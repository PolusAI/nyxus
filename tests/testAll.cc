#include <gtest/gtest.h>
#include "testDummy.h"
#include "test_gabor.h"
#include "test_download_data.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"

TEST(TEST_NYXUS, DUMMY_TEST){
  test_dummy_functiotestn();
}

TEST(TEST_GABOR_GPU, DSB2018){
  get("https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip", "dsb2018");
  #ifdef USE_GPU
    test_gabor_gpu_2018();
  #endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}