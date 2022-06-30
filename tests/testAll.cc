#include <gtest/gtest.h>
#include "testDummy.h"

TEST(TEST_NYXUS, DUMMY_TEST){
  test_dummy_function();
}

TEST(TEST_GABOR_GPU, DSB2018){
  test_gabor_gpu_2018();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}