#pragma once

#include <gtest/gtest.h>

//#include "../src/nyx/roi_cache.h"
//#include "../src/nyx/parallel.h"
//#include "../src/nyx/features/glcm.h"
//#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
//#include "test_data.h"
//#include "test_main_nyxus.h"

void test_roi_blacklist () 
{
    //=== Test global blacklists

    ASSERT_NO_THROW(Nyxus::theEnvironment.clear_roi_blacklist());

    std::string m;
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("", 123));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("xyz", 123));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("xyz", 456) == false);

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123,456", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("xyz", 123));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("xyz", 456));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("xyz", 789) == false);

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123,456,789", m));
    ASSERT_TRUE(m.empty());

    // Lexical error
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123,alphanumeric", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("alphanumeric,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123.45", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123,", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string(",123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("123,,456,789", m) == false);
    ASSERT_TRUE(m.empty() == false);

    //=== Test per-file blacklists

    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif: 123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123,456", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123,456,789", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:444,555,666", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:44,55,66;file3.tif:77,88,99", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file1.tif", 123));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file1.tif", 456));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file1.tif", 55) == false);
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file2.tif", 55));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file2.tif", 66));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file3.tif", 99));
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file3.tif", 55) == false);

    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1:123,44,55", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file1", 456) == false);
    ASSERT_TRUE(Nyxus::theEnvironment.roi_is_blacklisted("file1", 55));

    // Lexical error: bad file name
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1 file2.tif:123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123 alphanumeric stuff", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123,alphanumeric", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:alphanumeric,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1.tif:123.45", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1:123,", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1:,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(Nyxus::theEnvironment.parse_roi_blacklist_raw_string("file1:123,,456,789", m) == false);
    ASSERT_TRUE(m.empty() == false);
}
