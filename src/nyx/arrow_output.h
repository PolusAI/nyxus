#pragma once

#ifdef USE_ARROW
#include <string>
#include <memory>

#include "output_writers.h"
#include "helpers/helpers.h"

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
class ArrowOutput {

private:

    std::string arrow_file_path_ = "";
	std::string parquet_file_path_ = "";
	std::shared_ptr<ApacheArrowWriter> writer_ = nullptr;
	std::string arrow_output_type_ = "";


public:
    void create_arrow_file( const std::vector<std::string>& header,
                            const std::vector<std::string>& string_columns,
                            const std::vector<double>& results,
                            size_t num_rows,
                            const std::string& arrow_file_path="NyxusFeatures.arrow") {


        if(arrow_file_path != "" && !fs::is_directory(arrow_file_path) && !Nyxus::ends_with_substr(arrow_file_path, ".arrow")) {
            throw std::invalid_argument("The arrow file path must end in \".arrow\"");
        }

        if (arrow_file_path == "") {
            arrow_file_path_="NyxusFeatures.arrow";
        } else {
            arrow_file_path_ = arrow_file_path;
        }

        if (fs::is_directory(arrow_file_path)) {
            arrow_file_path_ += "/NyxusFeatures.arrow";
        }

        writer_ = WriterFactory::create_writer(arrow_file_path_);

        writer_->write(header, string_columns, results, num_rows);



    }

    void create_parquet_file(const std::vector<std::string>& header,
                             const std::vector<std::string>& string_columns,
                             const std::vector<double>& results,
                             size_t num_rows,
                             const std::string& parquet_file_path="NyxusFeatures.parquet") {

        if(parquet_file_path != "" && !fs::is_directory(parquet_file_path) && !Nyxus::ends_with_substr(parquet_file_path, ".parquet")) {
            throw std::invalid_argument("The parquet file path must end in \".parquet\"");
        }

        if (parquet_file_path == "") {
            parquet_file_path_="NyxusFeatures.parquet";
        } else {
            parquet_file_path_ = parquet_file_path;
        }

        if (fs::is_directory(parquet_file_path)) {
            parquet_file_path_ += "/NyxusFeatures.parquet";
        }

        writer_ = WriterFactory::create_writer(parquet_file_path_);

        writer_->write(header, string_columns, results, num_rows);
    }

    std::string get_arrow_file() {return arrow_file_path_;}

    std::string get_parquet_file() { return parquet_file_path_ ;}

    std::shared_ptr<arrow::Table> get_arrow_table(const std::vector<std::string>& header,
                                                      const std::vector<std::string>& string_columns,
                                                      const std::vector<double>& results,
                                                      size_t num_rows) {

        if (writer_ == nullptr) {
            writer_ = WriterFactory::create_writer("out.arrow");

            writer_->generate_arrow_table(header, string_columns, results, num_rows);

            return writer_->get_arrow_table();
        }

        auto table = writer_->get_arrow_table();

        return table;

    }

};
#endif
