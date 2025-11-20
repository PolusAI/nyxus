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

#include <sstream>
template <typename T>
bool tryParse(const std::string& str, T& result) 
{
    std::stringstream ss(str);
    ss >> result;
    return !ss.fail() && ss.eof(); // Check for failure and if entire string was consumed
}

void test_roi_blacklist_research()
{
    std::string m;
    Environment e;

    // induced error?

    e.clear_roi_blacklist();
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;

    //=== Test global blacklists

    e.clear_roi_blacklist();

    e.parse_roi_blacklist_raw_string("123", m);
//induced?
e.parse_roi_blacklist_raw_string("123", m);
e.parse_roi_blacklist_raw_string("123", m);
e.parse_roi_blacklist_raw_string("123", m);
    m.empty();

    e.roi_is_blacklisted("", 123);
    e.roi_is_blacklisted("xyz", 123);
    e.roi_is_blacklisted("xyz", 456) == false;
//induced?
e.roi_is_blacklisted("xyz", 123);
e.roi_is_blacklisted("xyz", 123);
e.roi_is_blacklisted("xyz", 123);
e.roi_is_blacklisted("xyz", 123);
e.parse_roi_blacklist_raw_string("1234", m);
e.parse_roi_blacklist_raw_string("12345", m);
e.parse_roi_blacklist_raw_string("12347", m);
e.parse_roi_blacklist_raw_string("1234587", m);

    m.empty();

    e.parse_roi_blacklist_raw_string("123,456", m);
//induced?
e.parse_roi_blacklist_raw_string("123,456", m);
e.parse_roi_blacklist_raw_string("123,456", m);
e.parse_roi_blacklist_raw_string("123,456", m);
e.parse_roi_blacklist_raw_string("123,456", m);
    m.empty();

    e.roi_is_blacklisted("xyz", 123);
    e.roi_is_blacklisted("xyz", 456);
    e.roi_is_blacklisted("xyz", 789) == false;
//induced?
e.roi_is_blacklisted("xyz", 789);
e.roi_is_blacklisted("xyz", 789);
e.roi_is_blacklisted("xyz", 789);
    m.empty();

//induced?
e.parse_roi_blacklist_raw_string("1233,4546", m);
e.parse_roi_blacklist_raw_string("12233,44356", m);
e.parse_roi_blacklist_raw_string("1523,4586", m);
e.parse_roi_blacklist_raw_string("155523,4576", m);


    e.parse_roi_blacklist_raw_string("123,456,789", m);
    m.empty();

//induced?
e.parse_roi_blacklist_raw_string("123,456,789", m);
e.parse_roi_blacklist_raw_string("123,456,789", m);
e.parse_roi_blacklist_raw_string("123,456,789", m);
e.parse_roi_blacklist_raw_string("123,456,789", m);


    // Lexical error

    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;

    // induced error?
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;
    e.parse_roi_blacklist_raw_string("123 alphanumeric stuff", m) == false;
    m.empty() == false;

    // Lexical error
    m.clear();
    e.parse_roi_blacklist_raw_string("123,alphanumeric", m) == false;
    m.empty() == false;

    // Lexical error
    m.clear();
    e.parse_roi_blacklist_raw_string("alphanumeric,123", m) == false;
    m.empty() == false;

    // Lexical error
    m.clear();
    e.parse_roi_blacklist_raw_string("123.45", m) == false;
    m.empty() == false;

    // Syntax error
    m.clear();
    e.parse_roi_blacklist_raw_string("123,", m) == false;
    m.empty() == false;

    // Syntax error
    m.clear();
    e.parse_roi_blacklist_raw_string(",123", m) == false;
    m.empty() == false;

    // Syntax error
    m.clear();
    e.parse_roi_blacklist_raw_string("123,,456,789", m) == false;
    m.empty() == false;

    //=== Test per-file blacklists

    m.clear();
    e.parse_roi_blacklist_raw_string("file1.tif:123", m);
    m.empty();

    e.parse_roi_blacklist_raw_string("file1.tif: 123", m);
    m.empty();

    e.parse_roi_blacklist_raw_string("file1.tif:123,456", m);
    m.empty();

    (e.parse_roi_blacklist_raw_string("file1.tif:123,456,789", m));
    (m.empty());

    (e.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:444,555,666", m));
    (m.empty());

    (e.parse_roi_blacklist_raw_string("file1.tif:123,456,789;file2.tif:44,55,66;file3.tif:77,88,99", m));
    (m.empty());

    (e.roi_is_blacklisted("file1.tif", 123));
    (e.roi_is_blacklisted("file1.tif", 456));
    (e.roi_is_blacklisted("file1.tif", 55) == false);
    (e.roi_is_blacklisted("file2.tif", 55));
    (e.roi_is_blacklisted("file2.tif", 66));
    (e.roi_is_blacklisted("file3.tif", 99));
    (e.roi_is_blacklisted("file3.tif", 55) == false);

    (e.parse_roi_blacklist_raw_string("file1:123,44,55", m));
    (m.empty());

    (e.roi_is_blacklisted("file1", 456) == false);
    (e.roi_is_blacklisted("file1", 55));

    // Lexical error: bad file name
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1 file2.tif:123", m) == false);
    (m.empty() == false);

    // Lexical error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1.tif:123 alphanumeric stuff", m) == false);
    (m.empty() == false);

    // Lexical error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1.tif:123,alphanumeric", m) == false);
    (m.empty() == false);

    // Lexical error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1.tif:alphanumeric,123", m) == false);
    (m.empty() == false);

    // Lexical error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1.tif:123.45", m) == false);
    (m.empty() == false);

    // Syntax error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1:123,", m) == false);
    (m.empty() == false);

    // Syntax error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1:,123", m) == false);
    (m.empty() == false);

    // Syntax error
    m.clear();
    (e.parse_roi_blacklist_raw_string("file1:123,,456,789", m) == false);
    (m.empty() == false);
}

