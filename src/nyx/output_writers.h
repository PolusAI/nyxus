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

    arrow::Status open(std::shared_ptr<arrow::io::RandomAccessFile> input, const std::string& file_path) {
        ARROW_ASSIGN_OR_RAISE(input, arrow::io::ReadableFile::Open(file_path));
    }

    arrow::Status open_parquet_file(std::shared_ptr<arrow::io::RandomAccessFile> input, arrow::MemoryPool* pool, std::unique_ptr<parquet::arrow::FileReader> arrow_reader) {
        ARROW_RETURN_NOT_OK(parquet::arrow::OpenFile(input, pool, &arrow_reader));
    }

    arrow::Status read_parquet_table(std::unique_ptr<parquet::arrow::FileReader> arrow_reader, std::shared_ptr<arrow::Table> table) {
        ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));
    }


public:

    /**
     * @brief Get the arrow table object
     * 
     * @return std::shared_ptr<arrow::Table> 
     */
    static std::shared_ptr<arrow::Table> get_arrow_table(const std::string& file_path) {

        auto file_extension = fs::path(file_path).extension().u8string();

        if (file_extension == ".parquet") {
            arrow::MemoryPool* pool = arrow::default_memory_pool();

            std::shared_ptr<arrow::io::RandomAccessFile> input;

            //auto status = this->open(input, file_path);
            input = arrow::io::ReadableFile::Open(file_path).ValueOrDie();
            
            std::unique_ptr<parquet::arrow::FileReader> arrow_reader;

            auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);

            if (!status.ok()) {
                    // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error reading Arrow file: " + err);
            }

            // Read entire file as a single Arrow table
            std::shared_ptr<arrow::Table> table;

            status = arrow_reader->ReadTable(&table);

            if (!status.ok()) {
                    // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error reading Arrow file: " + err);
            }

            return table;

        } else if (file_extension == ".arrow") {

            // Create a memory-mapped file for reading.
            std::shared_ptr<arrow::io::ReadableFile> input;
            input = arrow::io::ReadableFile::Open(file_path).ValueOrDie();

            // Create an IPC reader.
            //std::shared_ptr<arrow::ipc::RecordBatchFileReader> ipc_reader;
            auto ipc_reader = (arrow::ipc::RecordBatchStreamReader::Open(input.get())).ValueOrDie();

            return ipc_reader->ToTable().ValueOrDie();

            /*
            std::vector<std::shared_ptr<arrow::RecordBatch>> batches_array;


            arrow::ipc::RecordBatchFileReader * reader_ptr = ipc_reader.ValueOrDie().get()
            arrow::RecordBatchIterator ipc_reader_iterator = arrow::RecordBatchIterator(*reader_ptr);

            for (const auto& batch: ipc_reader_iterator) {
                batches_array.push_back(batch.ValueOrDie());
            }
            */

            /*
            for (int i = 0; i < ipc_reader->num_record_batches(); ++i) {
                std::shared_ptr<arrow::RecordBatch> batch = (ipc_reader->ReadRecordBatch(i)).ValueOrDie();

                batches_array.push_back(batch);
            }
            */
            
            //auto table = arrow::Table::FromRecordBatches(batches_array).ValueOrDie();

            //return table;
            
        } else {
            throw std::invalid_argument("Error: file must either be an Arrow or Parquet file.");
        }

    }

    /**
     * @brief Generate an Arrow table from Nyxus output
     * 
     * @param header Header data
     * @param string_columns String data
     * @param numeric_columns Numeric data
     * @param number_of_rows Number of rows
     * @return std::shared_ptr<arrow::Table> 
     */
    std::shared_ptr<arrow::Table> generate_arrow_table(const std::vector<std::string> &header,
                                                       const std::vector<std::string> &string_columns,
                                                       const std::vector<double> &numeric_columns,
                                                       int number_of_rows)
    {
        std::vector<std::shared_ptr<arrow::Field>> fields;

        fields.push_back(arrow::field(header[0], arrow::utf8()));
        fields.push_back(arrow::field(header[1], arrow::utf8()));
        fields.push_back(arrow::field(header[2], arrow::int32()));

        for (int i = 3; i < header.size(); ++i)
        {
            fields.push_back(arrow::field(header[i], arrow::float64()));
        }

        auto schema = arrow::schema(fields);

        arrow::StringBuilder string_builder_0;

        std::vector<std::string> temp_string_vec1(string_columns.size()/2);
        std::vector<std::string> temp_string_vec2(string_columns.size()/2);

        for (int i = 0; i < string_columns.size(); i+=2) {
            temp_string_vec1[i/2] = string_columns[i];
            temp_string_vec2[i/2] = string_columns[i+1];
        }
        
        PARQUET_THROW_NOT_OK(string_builder_0.AppendValues(temp_string_vec1));

        arrow::StringBuilder string_builder_1;
        
        PARQUET_THROW_NOT_OK(string_builder_1.AppendValues(temp_string_vec2));

        std::shared_ptr<arrow::Array> array_0, array_1;

        PARQUET_THROW_NOT_OK(string_builder_0.Finish(&array_0));
        PARQUET_THROW_NOT_OK(string_builder_1.Finish(&array_1));

        std::vector<std::shared_ptr<arrow::Array>> arrays;

        arrays.push_back(array_0);
        arrays.push_back(array_1);

        // add labels
        arrow::Int32Builder labels_builder;

        std::vector<int> temp_vec;
        int num_columns = numeric_columns.size() / number_of_rows;
        for (int i = 0; i < numeric_columns.size(); i += num_columns)
        {
            temp_vec.push_back(numeric_columns[i]);
        }


        PARQUET_THROW_NOT_OK(labels_builder.AppendValues(
            temp_vec));

        std::shared_ptr<arrow::Array> array_2;

        PARQUET_THROW_NOT_OK(labels_builder.Finish(&array_2));
        arrays.push_back(array_2);
        for (int i = 1; i < num_columns; ++i)
        {
            arrow::DoubleBuilder builder;

            std::vector<double> temp;

            for (int j = 0; j < number_of_rows; ++j)
            {
                temp.push_back(numeric_columns[i + (j * num_columns)]);
            }

            PARQUET_THROW_NOT_OK(builder.AppendValues(
                temp));

            std::shared_ptr<arrow::Array> temp_array;

            PARQUET_THROW_NOT_OK(builder.Finish(&temp_array));
            arrays.push_back(temp_array);
        }

        return arrow::Table::Make(schema, arrays);
    }

    /**
     * @brief Write Nyxus data to Arrow file
     * 
     * @param header Header data
     * @param string_columns String data
     * @param numeric_columns Numeric data
     * @param number_of_rows Number of rows
     * @return arrow::Status 
     */
    virtual arrow::Status write () = 0;

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

        arrow::Status setup(const std::vector<std::string> &header) {


            std::vector<std::shared_ptr<arrow::Field>> fields;
            

            fields.push_back(arrow::field(header[0], arrow::utf8()));
            fields.push_back(arrow::field(header[1], arrow::utf8()));
            fields.push_back(arrow::field(header[2], arrow::int64()));

            for (int i = 3; i < header.size(); ++i)
            {
                fields.push_back(arrow::field(header[i], arrow::float64()));
            }

            schema_ = arrow::schema(fields);

            PARQUET_ASSIGN_OR_THROW(
                output_stream_, arrow::io::FileOutputStream::Open(output_file_)
            );

            // Choose compression
            std::shared_ptr<parquet::WriterProperties> props =
                parquet::WriterProperties::Builder().compression(arrow::Compression::SNAPPY)->build();

            // Opt to store Arrow schema for easier reads back into Arrow
            std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
                parquet::ArrowWriterProperties::Builder().store_schema()->build();
            
            ARROW_ASSIGN_OR_RAISE(
                writer_, parquet::arrow::FileWriter::Open(*schema_,
                                                        arrow::default_memory_pool(), output_stream_,
                                                        props, arrow_props));

            return arrow::Status::OK();
        }

    public:

        ParquetWriter(const std::string& output_file, const std::vector<std::string>& header) : output_file_(output_file) {

            auto status = this->setup(header);

            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error writing setting up Arrow writer: " + err);
            }
        }

        /**
         * @brief Write to Parquet 
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numeric data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write () override {

            std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> features = Nyxus::get_feature_values();

            int num_rows = features.size();

            std::vector<std::shared_ptr<arrow::Array>> arrays;

            arrow::StringBuilder string_builder;
            std::shared_ptr<arrow::Array> intensity_array;


            arrow::Status append_status;
            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                append_status = string_builder.Append(std::get<0>(features[i])[0]);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = string_builder.Finish(&intensity_array);
            if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            

            arrays.push_back(intensity_array);
            string_builder.Reset();

            std::shared_ptr<arrow::Array> segmentation_array;

            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                append_status = string_builder.Append(std::get<0>(features[i])[1]);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = string_builder.Finish(&segmentation_array);

            if (!append_status.ok()) {
                // Handle read error
                auto err = append_status.ToString();
                throw std::runtime_error("Error writing Arrow file 2: " + err);
            }

            arrays.push_back(segmentation_array);

            arrow::Int64Builder int_builder;
            std::shared_ptr<arrow::Array> labels_array;
            // construct label column
            for (int i = 0; i < num_rows; ++i) {
                append_status = int_builder.Append(std::get<1>(features[i]));
                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = int_builder.Finish(&labels_array);
            if (!append_status.ok()) {
                // Handle read error
                auto err = append_status.ToString();
                throw std::runtime_error("Error writing Arrow file 2: " + err);
            }
            arrays.push_back(labels_array);

            // construct columns for each feature 
            for (int j = 0; j < std::get<2>(features[0]).size(); ++j) {

                arrow::DoubleBuilder builder;   
                std::shared_ptr<arrow::Array> double_array;

                for (int i = 0; i < num_rows; ++i) {
                    append_status = builder.Append(std::get<2>(features[i])[j]);

                    if (!append_status.ok()) {
                        // Handle read error
                        auto err = append_status.ToString();
                        throw std::runtime_error("Error writing Arrow file 2: " + err);
                    }
                }

                append_status =  builder.Finish(&double_array);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }

                arrays.push_back(double_array);
            }

            std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(schema_, num_rows, arrays);

            ARROW_ASSIGN_OR_RAISE(auto table,
                        arrow::Table::FromRecordBatches(schema_, {batch}));

            ARROW_RETURN_NOT_OK(writer_->WriteTable(*table.get(), batch->num_rows()));

            

            return arrow::Status::OK();
    }

    arrow::Status close () override {
        arrow::Status status = writer_->Close();

            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error closing the Arrow file: " + err);
            }
            return arrow::Status::OK();
            

        return arrow::Status::OK();
    }
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

        void set_schema(const std::vector<std::string> &header) {

            std::vector<std::shared_ptr<arrow::Field>> fields;

            fields.push_back(arrow::field(header[0], arrow::utf8()));
            fields.push_back(arrow::field(header[1], arrow::utf8()));
            fields.push_back(arrow::field(header[2], arrow::int64()));

            for (int i = 3; i < header.size(); ++i)
            {
                fields.push_back(arrow::field(header[i], arrow::float64()));
            }

            schema_ = arrow::schema(fields);

        }

        arrow::Status set_output_stream(const std::string& output_file){

            ARROW_ASSIGN_OR_RAISE(
                output_stream_, arrow::io::FileOutputStream::Open(output_file_)
            );

            return arrow::Status::OK();
        }

        void set_stream_writer() {
            writer_ = arrow::ipc::MakeStreamWriter(output_stream_,  schema_);
        }

        arrow::Status setup(const std::vector<std::string> &header) {


            std::vector<std::shared_ptr<arrow::Field>> fields;
            

            fields.push_back(arrow::field("intensity_image", arrow::utf8()));
            fields.push_back(arrow::field("segmentation_image", arrow::utf8()));
            fields.push_back(arrow::field("ROI_label", arrow::int64()));

            for (int i = 3; i < header.size(); ++i)
            {
                fields.push_back(arrow::field(header[i], arrow::float64()));
            }

            schema_ = arrow::schema(fields);

            ARROW_ASSIGN_OR_RAISE(
                output_stream_, arrow::io::FileOutputStream::Open(output_file_)
            );
            
            writer_ = arrow::ipc::MakeFileWriter(output_stream_,  schema_);

            return arrow::Status::OK();
        }

    public:

        ArrowIPCWriter(const std::string& output_file, const std::vector<std::string> &header) : output_file_(output_file) {

            auto status = this->setup(header);

            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error writing setting up Arrow writer: " + err);
            }
        }    

        /**
         * @brief Write to Arrow IPC
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numeric data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write () override {

            std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>> features = Nyxus::get_feature_values();

            int num_rows = features.size();

            std::vector<std::shared_ptr<arrow::Array>> arrays;

            arrow::StringBuilder string_builder;
            std::shared_ptr<arrow::Array> intensity_array;


            arrow::Status append_status;
            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                append_status = string_builder.Append(std::get<0>(features[i])[0]);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = string_builder.Finish(&intensity_array);
            if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            

            arrays.push_back(intensity_array);
            string_builder.Reset();

            std::shared_ptr<arrow::Array> segmentation_array;

            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                append_status = string_builder.Append(std::get<0>(features[i])[1]);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = string_builder.Finish(&segmentation_array);

            if (!append_status.ok()) {
                // Handle read error
                auto err = append_status.ToString();
                throw std::runtime_error("Error writing Arrow file 2: " + err);
            }

            arrays.push_back(segmentation_array);

            arrow::Int32Builder int_builder;
            std::shared_ptr<arrow::Array> labels_array;
            // construct label column
            for (int i = 0; i < num_rows; ++i) {
                append_status = int_builder.Append(std::get<1>(features[i]));
                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }

            append_status = int_builder.Finish(&labels_array);
            if (!append_status.ok()) {
                // Handle read error
                auto err = append_status.ToString();
                throw std::runtime_error("Error writing Arrow file 2: " + err);
            }
            arrays.push_back(labels_array);

            // construct columns for each feature 
            for (int j = 0; j < std::get<2>(features[0]).size(); ++j) {

                arrow::DoubleBuilder builder;   
                std::shared_ptr<arrow::Array> double_array;

                for (int i = 0; i < num_rows; ++i) {
                    append_status = builder.Append(std::get<2>(features[i])[j]);

                    if (!append_status.ok()) {
                        // Handle read error
                        auto err = append_status.ToString();
                        throw std::runtime_error("Error writing Arrow file 2: " + err);
                    }
                }

                append_status =  builder.Finish(&double_array);

                if (!append_status.ok()) {
                    // Handle read error
                    auto err = append_status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }

                arrays.push_back(double_array);
            }

            std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(schema_, num_rows, arrays);

            auto status = writer_->get()->WriteRecordBatch(*batch);

            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error writing Arrow file 2: " + err);
            }

            return arrow::Status::OK();
        }


        arrow::Status close () {

            arrow::Status status = writer_->get()->Close();

            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error closing the Arrow file: " + err);
            }
            return arrow::Status::OK();
            
        }
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
        static std::shared_ptr<ApacheArrowWriter> create_writer(const std::string &output_file, const std::vector<std::string> &header) {
            
            if (Nyxus::ends_with_substr(output_file, ".parquet")) {
                
                return std::make_shared<ParquetWriter>(output_file, header);

            } else if (Nyxus::ends_with_substr(output_file, ".arrow") || Nyxus::ends_with_substr(output_file, ".feather")) {
                
                return std::make_shared<ArrowIPCWriter>(output_file, header);

            } else {

                std::filesystem::path path(output_file);

                if (path.has_extension()) {
                    std::string file_extension = path.extension().string();
                    
                    throw std::invalid_argument("No writer option for extension \"" + file_extension + "\". Valid options are \".parquet\" or \".arrow\".");
 
                } else {

                    throw std::invalid_argument("No extension type was provided in the path. ");

                }
            }
        }
};


#endif