#pragma once

#include <gtest/gtest.h>
#include "../src/nyx/environment.h"

void test_roi_blacklist () 
{
    //=== Test global blacklists

    Environment e;
    ASSERT_NO_THROW(e.clear_roi_blacklist());

    std::string m;
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.roi_is_blacklisted("", 123));
    ASSERT_TRUE(e.roi_is_blacklisted("xyz", 123));
    ASSERT_TRUE(e.roi_is_blacklisted("xyz", 456) == false);
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123,456", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.roi_is_blacklisted("xyz", 123));
    ASSERT_TRUE(e.roi_is_blacklisted("xyz", 456));
    ASSERT_TRUE(e.roi_is_blacklisted("xyz", 789) == false);
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123,456,789", m));
    ASSERT_TRUE(m.empty());

    // Lexical error
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123,alphanumeric", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("alphanumeric,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123.45", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123,", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string(",123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("123,,456,789", m) == false);
    ASSERT_TRUE(m.empty() == false);

    //=== Test per-file blacklists

    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif: 123", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123,456", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123,456,789", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:444,555,666", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:44,55,66;file3.tif:77,88,99", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.roi_is_blacklisted("file1.tif", 123));
    ASSERT_TRUE(e.roi_is_blacklisted("file1.tif", 456));
    ASSERT_TRUE(e.roi_is_blacklisted("file1.tif", 55) == false);
    ASSERT_TRUE(e.roi_is_blacklisted("file2.tif", 55));
    ASSERT_TRUE(e.roi_is_blacklisted("file2.tif", 66));
    ASSERT_TRUE(e.roi_is_blacklisted("file3.tif", 99));
    ASSERT_TRUE(e.roi_is_blacklisted("file3.tif", 55) == false);

    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1:123,44,55", m));
    ASSERT_TRUE(m.empty());

    ASSERT_TRUE(e.roi_is_blacklisted("file1", 456) == false);
    ASSERT_TRUE(e.roi_is_blacklisted("file1", 55));

    // Lexical error: bad file name
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1 file2.tif:123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123 alphanumeric stuff", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123,alphanumeric", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:alphanumeric,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Lexical error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1.tif:123.45", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1:123,", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1:,123", m) == false);
    ASSERT_TRUE(m.empty() == false);

    // Syntax error
    m.clear();
    ASSERT_TRUE(e.parse_roi_blacklist_raw_string("file1:123,,456,789", m) == false);
    ASSERT_TRUE(m.empty() == false);
}

// delete old unused exploratory test code that was left behind.

