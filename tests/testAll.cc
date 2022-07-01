#include <gtest/gtest.h>
#include "testDummy.h"
#include "test_gabor.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"

TEST(TEST_NYXUS, DUMMY_TEST){
  test_dummy_function();
}

TEST(TEST_GABOR_GPU, DSB2018){
  #ifdef USE_GPU
    if(theEnvironment.using_gpu()) test_gabor_gpu_2018();
  #endif
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}