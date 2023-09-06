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
#include "globals.h"

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
    virtual arrow::Status write (const std::string & intFpath, const std::string & segFpath, const std::string & outputDir) = 0;

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
        arrow::Status write (const std::string & intFpath, const std::string & segFpath, const std::string & outputDir) override {
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
        std::shared_ptr<arrow::Schema> schema_;
        std::shared_ptr<arrow::io::FileOutputStream> output_stream_;
        arrow::Result<std::shared_ptr<arrow::ipc::RecordBatchWriter>> writer_;

        void set_schema(const std::vector<std::string> &header) {

            std::vector<std::shared_ptr<arrow::Field>> fields;

            fields.push_back(arrow::field("intensity_image", arrow::utf8()));
            fields.push_back(arrow::field("segmentation_image", arrow::utf8()));
            fields.push_back(arrow::field("ROI_label", arrow::int64()));

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

    public:

        ArrowIPCWriter(const std::string& output_file, const std::vector<std::string> &header) : output_file_(output_file) {

            this->set_schema(header);

            this->set_output_stream(output_file);

            this->set_stream_writer();
        }    

        /**
         * @brief Write to Arrow IPC
         * 
         * @param header Header data
         * @param string_columns String data (filenames)
         * @param numeric_columns Numberic data (feature calculations)
         * @param number_of_rows Number of rows
         * @return arrow::Status 
         */
        arrow::Status write (const std::string & intFpath, const std::string & segFpath, const std::string & outputDir) override {

            std::shared_ptr<arrow::RecordBatch> batch;

            auto features = Nyxus::get_feature_values();

            int num_rows = features.size();

            std::vector<std::shared_ptr<arrow::Array>> arrays;

            arrow::StringBuilder string_builder;
            std::shared_ptr<arrow::Array> intensity_array;

            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                string_builder.Append(std::get<0>(features[i])[1]);
            }

            string_builder.Finish(&intensity_array);

            arrays.push_back(intensity_array);
            string_builder.Reset();

            std::shared_ptr<arrow::Array> segmentation_array;

            // construct intensity column
            for (int i = 0; i < num_rows; ++i) {
                string_builder.Append(std::get<0>(features[i])[0]);
            }

            string_builder.Finish(&segmentation_array);

            arrays.push_back(segmentation_array);

            arrow::Int32Builder int_builder;
            std::shared_ptr<arrow::Array> labels_array;
            // construct label column
            for (int i = 0; i < num_rows; ++i) {
                int_builder.Append(std::get<1>(features[i]));
            }

            int_builder.Finish(&labels_array);

            arrays.push_back(labels_array);

            // construct columns for each feature 
            for (int j = 3; j < std::get<2>(features[0]).size(); ++j) {

                arrow::DoubleBuilder builder;   
                std::shared_ptr<arrow::Array> double_array;

                for (int i = 0; i < num_rows; ++i) {
                    builder.Append(std::get<2>(features[i])[j]);
                }

                builder.Finish(&double_array);

                arrays.push_back(double_array);
            }


            //ARROW_ASSIGN_OR_RAISE(
            //    batch, arrow::RecordBatch::Make(schema_, num_rows, arrays)
            //);

            batch = arrow::RecordBatch::Make(schema_, num_rows, arrays);

            // put data in record batch now or make a non empty batch

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
                std::cout << "creating parquet file" << std::endl;
                return std::make_shared<ParquetWriter>(output_file);

            } else if (Nyxus::ends_with_substr(output_file, ".arrow") || Nyxus::ends_with_substr(output_file, ".feather")) {
                std::cout << "creating arrow file" << std::endl;
                
                return std::make_shared<ArrowIPCWriter>(output_file, header);

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