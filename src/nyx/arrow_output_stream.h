#pragma once

#include <optional>
#include <string>
#include <memory>
#include <tuple>
#include <vector>

#include "output_writers.h"
#include "save_option.h"

#ifdef USE_ARROW
#include <arrow/table.h>

/**
 * @brief Class to write to Apache Arrow formats
 * 
 * This class provides methods for writing to the Arrow IPC and Parquet formats.
 * 
 */
class ArrowOutputStream {

private:

  std::string arrow_file_path_ = "";
	std::unique_ptr<ApacheArrowWriter> writer_ = nullptr;
	std::string arrow_output_type_ = "";
  std::shared_ptr<arrow::Table> arrow_table_ = nullptr;

public:
    std::tuple<bool, std::optional<std::string>> create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header);
    std::shared_ptr<arrow::Table> get_arrow_table(const std::string& file_path);
    std::string get_arrow_path();
    std::tuple<bool, std::optional<std::string>> write_arrow_file (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features);
    std::tuple<bool, std::optional<std::string>> close_arrow_file ();
};

#else 

/**
 * @brief Class to write to Apache Arrow formats
 * 
 * This class provides a place holder for the Arrow writer class when Nyxus is built without arrow.
 * 
 */
class ArrowOutputStream {

public:
    std::tuple<bool, std::optional<std::string>> create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header);
    std::tuple<bool, std::optional<std::string>> write_arrow_file (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features);
    std::tuple<bool, std::optional<std::string>> close_arrow_file ();
};


#endif