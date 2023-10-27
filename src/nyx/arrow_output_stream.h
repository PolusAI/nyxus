#pragma once

#include <string>
#include <memory>

#include "output_writers.h"
#include "helpers/helpers.h"
#include "save_option.h"

#ifdef USE_ARROW
#include <arrow/table.h>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

/**
 * @brief Class to write to Apache Arrow formats
 * 
 * This class provides methods for writing to the Arrow IPC and Parquet formats.
 * 
 */
class ArrowOutputStream {

private:

  std::string arrow_file_path_ = "";
	std::shared_ptr<ApacheArrowWriter> writer_ = nullptr;
	std::string arrow_output_type_ = "";
  std::shared_ptr<arrow::Table> arrow_table_ = nullptr;

public:
    std::shared_ptr<ApacheArrowWriter> create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header);
    std::shared_ptr<arrow::Table> get_arrow_table(const std::string& file_path);
    std::string get_arrow_path();
};

#else 

// Replace arrow::Table with a dummy variable
namespace arrow {
  using Table = bool;
};

/**
 * @brief Class to write to Apache Arrow formats
 * 
 * This class provides a place holder for the Arrow writer class when Nyxus is built without arrow.
 * 
 */
class ArrowOutputStream {

public:
    std::shared_ptr<ApacheArrowWriter> create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header);
    std::shared_ptr<arrow::Table> get_arrow_table(const std::string& file_path);
    std::string get_arrow_path();
};


#endif