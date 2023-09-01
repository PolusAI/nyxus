#pragma once

#ifdef USE_ARROW
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <arrow/ipc/api.h>
#include <arrow/result.h>

#include <arrow/csv/api.h>

#include <vector>
#include <string>
#include <filesystem> 
#include <stdexcept>
#include <memory>

#include "helpers/helpers.h"

#include <iostream>



/**
 * @brief Base class for creating Apache Arrow output writers
 * 
 * This class provides methods for the Arrow table used for writing to Arrow formats and
 * provides virtual functions to overriden for writing to different formats
 * 
 */
class ApacheArrowWriter
{
protected:
    std::shared_ptr<arrow::Table> table_;
public:

    /**
     * @brief Get the arrow table object
     * 
     * @return std::shared_ptr<arrow::Table> 
     */
    std::shared_ptr<arrow::Table> get_arrow_table() {return table_;}

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

        int idx;
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

        //table_ = arrow::Table::Make(schema, arrays);
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
    virtual arrow::Status write (const std::string& csv_path, const std::vector<std::string> &header) = 0;

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

    public:

        ParquetWriter(const std::string& output_file) : output_file_(output_file) {}

        /**
         * @brief Write to Parquet 
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numberic data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write (const std::string& csv_path, const std::vector<std::string> &header) override {
            /*
            table_ = generate_arrow_table(header, string_columns, numeric_columns, number_of_rows);

            std::shared_ptr<arrow::io::FileOutputStream> outfile;

            PARQUET_ASSIGN_OR_THROW(
                outfile, arrow::io::FileOutputStream::Open(output_file_));

            PARQUET_THROW_NOT_OK(
                parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(), outfile, 3));
            */ 

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

    public:

        ArrowIPCWriter(const std::string& output_file) : output_file_(output_file) {}    

        /**
         * @brief Write to Arrow IPC
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numberic data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write (const std::string& csv_path, const std::vector<std::string> &header) override {
            std::cout << "calling write method" << std::endl;
            arrow::io::IOContext io_context = arrow::io::default_io_context();
            std::shared_ptr<arrow::io::InputStream> input;
            std::cout << "csv path: " << csv_path << std::endl;
            ARROW_ASSIGN_OR_RAISE (input, arrow::io::ReadableFile::Open(csv_path));

            auto read_options = arrow::csv::ReadOptions::Defaults();
            auto parse_options = arrow::csv::ParseOptions::Defaults();
            auto convert_options = arrow::csv::ConvertOptions::Defaults();

            std::cout << "1" << std::endl;

            // Instantiate StreamingReader from input stream and options
            auto maybe_reader =
                arrow::csv::StreamingReader::Make(io_context,
                                                input,
                                                read_options,
                                                parse_options,
                                                convert_options);
            if (!maybe_reader.ok()) {
                // Handle StreamingReader instantiation error...
                throw std::runtime_error("Error initializing the reader.");
            }
            std::shared_ptr<arrow::csv::StreamingReader> reader = *maybe_reader;
            std::cout << "2" << std::endl;


            // Create the Arrow file writer
            std::shared_ptr<arrow::io::FileOutputStream> output_stream;

            std::cout << "3" << std::endl;

            ARROW_ASSIGN_OR_RAISE(
                output_stream, arrow::io::FileOutputStream::Open(output_file_)
            );

            std::vector<std::shared_ptr<arrow::Field>> fields;

            fields.push_back(arrow::field("intensity_image", arrow::utf8()));
            fields.push_back(arrow::field("segmentation_image", arrow::utf8()));
            fields.push_back(arrow::field("ROI_label", arrow::int64()));

            for (int i = 3; i < header.size(); ++i)
            {
                fields.push_back(arrow::field(header[i], arrow::float64()));
            }

            auto schema = arrow::schema(fields);

            // Set aside a RecordBatch pointer for re-use while streaming
            //std::shared_ptr<arrow::RecordBatch> batch;
            
            //ARROW_ASSIGN_OR_RAISE(
            //    batch, arrow::RecordBatch::MakeEmpty(schema)
            //);

            auto writer = arrow::ipc::MakeStreamWriter(output_stream,  schema);

            std::cout << "before writing" << std::endl;
            while (true) {
                std::shared_ptr<arrow::RecordBatch> temp_batch;
                ARROW_ASSIGN_OR_RAISE(
                    temp_batch, arrow::RecordBatch::MakeEmpty(schema)
                );
                // Attempt to read the first RecordBatch
                std::cout << "reading csv" << std::endl;
                arrow::Status status = reader->ReadNext(&temp_batch);

                //std::cout << batch->ToString() << std::endl;

                if (!status.ok()) {
                    // Handle read error
                    throw std::runtime_error("Error writing Arrow file 1.");
                }

                if (temp_batch == NULL) {
                    // Handle end of file
                    break;
                }

                //std::cout << temp_batch->schema()->ToString() << std::endl;
                //std::cout << "-----------" << std::endl;
                //std::cout << schema->ToString() << std::endl;

                // Check if the batch schema matches the writer schema
                if (!temp_batch->schema()->Equals(schema)) {

                    // Schemas are different, identify the differences
                    auto desired_fields = temp_batch->schema()->fields();
                    auto actual_fields = schema->fields();

                    for (size_t i = 0; i < desired_fields.size(); ++i) {
                        if (!desired_fields[i]->Equals(actual_fields[i])) {
                            std::cout << "Field " << i << " differs:" << std::endl;
                            std::cout << "Desired Field: " << desired_fields[i]->ToString() << std::endl;
                            std::cout << "Actual Field: " << actual_fields[i]->ToString() << std::endl;
                        }
                    }


                    // Handle schema mismatch (e.g., modify schema to match batch schema)
                    // You might need to adapt the schema or handle this situation based on your use case.
                    throw std::runtime_error("Error: the schemas do not match");
                }

                status = writer->get()->WriteRecordBatch(*temp_batch);
                if (!status.ok()) {
                    // Handle read error
                    auto err = status.ToString();
                    throw std::runtime_error("Error writing Arrow file 2: " + err);
                }
            }
            std::cout << "Closing Arrow file" << std::endl;
            arrow::Status status = writer->get()->Close();
            std::cout << "arrow file closed" << std::endl;
            if (!status.ok()) {
                // Handle read error
                auto err = status.ToString();
                throw std::runtime_error("Error closing the Arrow file: " + err);
            }
            return arrow::Status::OK();
            /*
            // Create Arrow schema from CSV schema
            auto csv_schema = arrow::schema({ });
            
            // Create Arrow IPC file writer
            std::shared_ptr<arrow::io::FileOutputStream> output_stream;
            auto status = arrow::io::FileOutputStream::Open(output_file_, &output_stream);
            if (!status.ok()) {
                // Handle output file creation error
                throw std::runtime_error("Error creating Arrow file.");
            }
            auto ipc_writer = arrow::ipc::MakeStreamWriter(output_stream.get(), csv_schema);
            
            // Read CSV file row by row
            std::shared_ptr<arrow::csv::TableReader> csv_reader;
            auto status2 = arrow::csv::TableReader::Make(arrow::default_memory_pool(), csv_path, arrow::csv::ReadOptions::Defaults(), csv_schema, &csv_reader);
            if (!status.ok()) {
                // Handle CSV reader creation error
                throw std::runtime_error("Error creating Arrow file.");
            }
            
            while (true) {
                std::shared_ptr<arrow::Table> table;
                status = csv_reader->Read(&table);
                if (!status.ok() || table->num_rows() == 0) {
                    // Either error or end of file
                    break;
                }
                
                // Write the current batch of rows to Arrow IPC file
                status = ipc_writer->WriteTable(*table);
                if (!status.ok()) {
                    // Handle IPC writing error
                    return status;
                }
            }
            
            // Close IPC writer and output stream
            ipc_writer->Close();
            output_stream->Close();

            return arrow::Status::OK();
            */
            /*
            table_ = generate_arrow_table(header, string_columns, numeric_columns, number_of_rows);

            // Create the Arrow file writer
            std::shared_ptr<arrow::io::FileOutputStream> output_stream;

            ARROW_ASSIGN_OR_RAISE(
                output_stream, arrow::io::FileOutputStream::Open(output_file_)
            );

            auto writer = arrow::ipc::MakeFileWriter(output_stream, table_->schema());

            // Write the Arrow table to file
            writer->get()->WriteTable(*table_);
            writer->get()->Close();


            */
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
        static std::shared_ptr<ApacheArrowWriter> create_writer(const std::string &output_file) {
            
            if (Nyxus::ends_with_substr(output_file, ".parquet")) {
                std::cout << "creating parquet file" << std::endl;
                return std::make_shared<ParquetWriter>(output_file);

            } else if (Nyxus::ends_with_substr(output_file, ".arrow") || Nyxus::ends_with_substr(output_file, ".feather")) {
                std::cout << "creating arrow file" << std::endl;
                
                return std::make_shared<ArrowIPCWriter>(output_file);

            } else {
                std::cout << "error branch" << std::endl;
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