#include <gtest/gtest.h>
#include "testDummy.h"

TEST(TEST_NYXUS, DUMMY_TEST){
  test_dummy_function();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}