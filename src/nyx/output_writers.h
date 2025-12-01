#pragma once

#define FTABLE_RECORD std::tuple<std::vector<std::string>,int,double,std::vector<double>>
#define FTABLE_INTSEG   0
#define FTABLE_ROILBL   1
#define FTABLE_TIMEPOS  2
#define FTABLE_FBEGIN   3
#define FTABLE_SAFENAN  -0.0

#ifdef USE_ARROW

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <optional>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/writer.h>

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
     * @brief Write Nyxus data to Arrow file
     *
     * @param header Header data
     * @param string_columns String data
     * @param numeric_columns Numeric data
     * @param number_of_rows Number of rows
     * @return arrow::Status
     */

    virtual arrow::Status write(const std::vector<FTABLE_RECORD>& features) = 0;

    virtual arrow::Status close() = 0;

    virtual ~ApacheArrowWriter() = default;

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

    arrow::Status setup(const std::vector<std::string>& header);

public:

    ParquetWriter(const std::string& output_file, const std::vector<std::string>& header);

    arrow::Status write(const std::vector<FTABLE_RECORD>& features) override;

    arrow::Status close() override;
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

    arrow::Status setup(const std::vector<std::string>& header);

public:

    ArrowIPCWriter(const std::string& output_file, const std::vector<std::string>& header);

    /**
     * @brief Write to Arrow IPC
     *
     * @param header Header data
     * @param string_columns String data (filenames)
     * @param numeric_columns Numeric data (feature calculations)
     * @param number_of_rows Number of rows
     * @return arrow::Status
     */

    arrow::Status write(const std::vector<FTABLE_RECORD>& features) override;

    arrow::Status close() override;
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
     * @return std::unique_ptr<ApacheArrowWriter>
     */
    static std::tuple<std::unique_ptr<ApacheArrowWriter>, std::optional<std::string>> create_writer(const std::string& output_file, const std::vector<std::string>& header);
};

#endif
