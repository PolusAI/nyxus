#pragma once

#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <arrow/ipc/api.h>
#include <arrow/result.h>
#include <arrow/ipc/reader.h>

#include <arrow/csv/api.h>

#include <vector>
#include <string>
#include <filesystem> 
#include <stdexcept>
#include <memory>

#include "helpers/helpers.h"
#include "globals.h"

#include <iostream>

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
 * @brief Base class for creating Apache Arrow output writers
 * 
 * This class provides methods for the Arrow table used for writing to Arrow formats and
 * provides virtual functions to overridden for writing to different formats
 * 
 */
class ApacheArrowWriter
{

private: 
    std::shared_ptr<arrow::Table> table_ = nullptr;

public:

    /**
     * @brief Get the arrow table object
     * 
     * @return std::shared_ptr<arrow::Table> 
     */
    std::shared_ptr<arrow::Table> get_arrow_table(const std::string& file_path);

    /**
     * @brief Write Nyxus data to Arrow file
     * 
     * @param header Header data
     * @param string_columns String data
     * @param numeric_columns Numeric data
     * @param number_of_rows Number of rows
     * @return arrow::Status 
     */
    virtual arrow::Status write (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features) = 0;

    virtual arrow::Status close () = 0;

};

/**
 * @brief Class to write to Parquet format
 * 
 * Extends ApacheArrowWriter class and implements write method for the Parquet format.
 * 
 */
class ParquetWriter : public ApacheArrowWriter {
    private:

        std::string output_file_;
        std::shared_ptr<arrow::Schema> schema_;
        std::shared_ptr<arrow::io::FileOutputStream> output_stream_;
        std::unique_ptr<parquet::arrow::FileWriter> writer_;

        arrow::Status setup(const std::vector<std::string> &header);

    public:

        ParquetWriter(const std::string& output_file, const std::vector<std::string>& header);

        
        arrow::Status write (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features) override;

        arrow::Status close () override;
};

/**
 * @brief Write to Apache IPC format (feather)
 * 
 * Extends ApacheArrowWriter and overrides write method for Arrow IPC format
 * 
 */
class ArrowIPCWriter : public ApacheArrowWriter {
    private:

        std::string output_file_;
        std::shared_ptr<arrow::Schema> schema_;
        std::shared_ptr<arrow::io::FileOutputStream> output_stream_;
        arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writer_;

        arrow::Status setup(const std::vector<std::string> &header);

    public:

        ArrowIPCWriter(const std::string& output_file, const std::vector<std::string> &header);

        /**
         * @brief Write to Arrow IPC
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numeric data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features) override;


        arrow::Status close () override;
};

/**
 * @brief Factory to create an ApacheArrowWriter based on type of file pass
 * 
 */
class WriterFactory {

    public:

        /**
         * @brief Create an ApacheArrowWriter based on the type of file passed.
         * 
         * @param output_file Path to output file (.arrow or .parquet)
         * @return std::shared_ptr<ApacheArrowWriter> 
         */
        static std::shared_ptr<ApacheArrowWriter> create_writer(const std::string &output_file, const std::vector<std::string> &header);
};


#endif