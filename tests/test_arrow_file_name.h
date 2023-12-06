#pragma once

#include <gtest/gtest.h>
#include <string>
#include "../src/nyx/globals.h"
#include "../src/nyx/save_option.h"

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif


/*
            output_path			condition			verdict
Case 1: 	/foo/bar		exist in fs				is a directory, append default filename with proper ext
            /foo/bar/		or ends with / or \
            \foo\bar\			

Case 2:		/foo/bar		does not exist in fs	assume the extension is missing, append proper ext
                            but /foo exists

Case 3: 	/foo/bar		neither /foo nor 		treat as directory, append default filename with proper ext
                            /foo/bar exists in fs

Case 4: 	/foo/bar.ext	exists in fs and is a 	append default filename with proper ext
                            directory	
        
Case 5: 	/foo/bar.ext	does not exist in fs  	this is a file, check if ext is correct and modify if needed

Case 6:		empty									default filename with proper ext
*/

void test_file_naming_case_1 (const fs::path& temp_dir);
void test_file_naming_case_2 (const fs::path& temp_dir);
void test_file_naming_case_3 (const fs::path& temp_dir);
void test_file_naming_case_4 (const fs::path& temp_dir);
void test_file_naming_case_5 (const fs::path& temp_dir);
void test_file_naming_case_6 (const fs::path& temp_dir);

void test_file_naming () {
    auto temp = fs::temp_directory_path() / "nyxus_temp/";

    if (!fs::exists(temp)) {
        auto is_created = fs::create_directory(temp);

        if (!is_created) {
            FAIL() << "Error creating temp directory for nyxus tests." << std::endl;
        }
    }

    test_file_naming_case_1(temp);
    test_file_naming_case_2(temp);
    test_file_naming_case_3(temp);
    test_file_naming_case_4(temp);
    test_file_naming_case_5(temp);
    test_file_naming_case_6(temp);

    auto num_deleted = fs::remove_all(temp);

    if (num_deleted == 0) {
        std::cerr << "WARNING: Temporary directory at " << temp.u8string() << " could not be deleted." << std::endl;
    }
}

void test_file_naming_case_1 (const fs::path& temp_dir) {

    fs::path output_path = temp_dir/"foo1/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 1." << std::endl;
        }
    }

    output_path = output_path / "bar1/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 1." << std::endl;
        }
    }

    auto result = get_arrow_filename(output_path.u8string(), "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = temp_dir.u8string() + "foo1/bar1/NyxusFeatures.arrow";

    ASSERT_TRUE(result == expected);
}

void test_file_naming_case_2 (const fs::path& temp_dir) {

    fs::path output_path = temp_dir / "foo2/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 2." << std::endl;
        }
    }

    output_path = output_path / "bar2";

    auto result = get_arrow_filename(output_path.u8string(), "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = temp_dir.u8string() + "foo2/bar2.arrow";

    ASSERT_TRUE(result == expected);
}

void test_file_naming_case_3 (const fs::path& temp_dir) {
    fs::path output_path = temp_dir/"foo3/bar3/";

    auto result = get_arrow_filename(output_path.u8string(), "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = temp_dir.u8string() + "foo3/bar3/NyxusFeatures.arrow";

    ASSERT_TRUE(result == expected);
}

void test_file_naming_case_4 (const fs::path& temp_dir) {

    fs::path output_path = temp_dir / "foo4/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 4." << std::endl;
        }
    }

    output_path = output_path / "bar4.ext/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 4." << std::endl;
        }
    }

    auto result = get_arrow_filename(output_path.u8string(), "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = temp_dir.u8string() + "foo4/bar4.ext/NyxusFeatures.arrow";

    ASSERT_TRUE(result == expected);
    
}

void test_file_naming_case_5 (const fs::path& temp_dir) {
    fs::path output_path = temp_dir / "foo4/";

    if (!fs::exists(output_path)) {
        auto is_created = fs::create_directory(output_path);

        if (!is_created) {
            FAIL() << "Error creating directory for case 2." << std::endl;
        }
    }

    output_path = output_path / "bar4.arrow";

    auto result = get_arrow_filename(output_path.u8string(), "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = temp_dir.u8string() + "foo4/bar4.arrow";

    ASSERT_TRUE(result == expected);
}

void test_file_naming_case_6 (const fs::path& temp_dir) {

    auto result = get_arrow_filename("", "NyxusFeatures", SaveOption::saveArrowIPC);

    std::string expected = "NyxusFeatures.arrow";

    ASSERT_TRUE(result == expected);
}