#include "output_writers.h"

#ifdef USE_ARROW

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

#include <iostream>
#include <parquet/arrow/reader.h>

#include "helpers/helpers.h"


arrow::Status ParquetWriter::setup(const std::vector<std::string> &header) {


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



ParquetWriter::ParquetWriter(const std::string& output_file, const std::vector<std::string>& header) : output_file_(output_file) {

    auto status = this->setup(header);

    if (!status.ok()) {
        // Handle read error
        std::cout << "Error writing setting up Arrow writer: " << status.ToString() << std::endl;
    }
}

        
arrow::Status ParquetWriter::write (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features) {

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
            return append_status;
        }
    }

    append_status = string_builder.Finish(&intensity_array);
    if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    

    arrays.push_back(intensity_array);
    string_builder.Reset();

    std::shared_ptr<arrow::Array> segmentation_array;

    // construct intensity column
    for (int i = 0; i < num_rows; ++i) {
        append_status = string_builder.Append(std::get<0>(features[i])[1]);

        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    }

    append_status = string_builder.Finish(&segmentation_array);

    if (!append_status.ok()) {
        // Handle read error
        return append_status;
    }

    arrays.push_back(segmentation_array);

    arrow::Int64Builder int_builder;
    std::shared_ptr<arrow::Array> labels_array;
    // construct label column
    for (int i = 0; i < num_rows; ++i) {
        append_status = int_builder.Append(std::get<1>(features[i]));
        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    }

    append_status = int_builder.Finish(&labels_array);
    if (!append_status.ok()) {
        // Handle read error
        return append_status;
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
            return append_status;
            }
        }

        append_status =  builder.Finish(&double_array);

        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }

        arrays.push_back(double_array);
    }

    std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(schema_, num_rows, arrays);

    ARROW_ASSIGN_OR_RAISE(auto table,
                arrow::Table::FromRecordBatches(schema_, {batch}));


    std::cout << table->ToString() << std::endl;

    ARROW_RETURN_NOT_OK(writer_->WriteTable(*table.get(), batch->num_rows()));

    

    return arrow::Status::OK();
}

arrow::Status ParquetWriter::close () {
    arrow::Status status = writer_->Close();

        if (!status.ok()) {
            // Handle read error
            return status; 
        }
        return arrow::Status::OK();
        

    return arrow::Status::OK();
}

arrow::Status ArrowIPCWriter::setup(const std::vector<std::string> &header) {

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



ArrowIPCWriter::ArrowIPCWriter(const std::string& output_file, const std::vector<std::string> &header) : output_file_(output_file) {

    auto status = this->setup(header);

}    


arrow::Status ArrowIPCWriter::write (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features) {


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
            return append_status;
        }
    }

    append_status = string_builder.Finish(&intensity_array);
    if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    

    arrays.push_back(intensity_array);
    string_builder.Reset();

    std::shared_ptr<arrow::Array> segmentation_array;

    // construct intensity column
    for (int i = 0; i < num_rows; ++i) {
        append_status = string_builder.Append(std::get<0>(features[i])[1]);

        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    }

    append_status = string_builder.Finish(&segmentation_array);

    if (!append_status.ok()) {
        // Handle read error
        return append_status;
    }

    arrays.push_back(segmentation_array);

    arrow::Int32Builder int_builder;
    std::shared_ptr<arrow::Array> labels_array;
    // construct label column
    for (int i = 0; i < num_rows; ++i) {
        append_status = int_builder.Append(std::get<1>(features[i]));
        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }
    }

    append_status = int_builder.Finish(&labels_array);
    if (!append_status.ok()) {
        // Handle read error
        return append_status;
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
                return append_status;
            }
        }

        append_status =  builder.Finish(&double_array);

        if (!append_status.ok()) {
            // Handle read error
            return append_status;
        }

        arrays.push_back(double_array);
    }

    std::shared_ptr<arrow::RecordBatch> batch = arrow::RecordBatch::Make(schema_, num_rows, arrays);

    auto status = writer_->get()->WriteRecordBatch(*batch);

    if (!status.ok()) {
        // Handle read error
        return status;
    }

    return arrow::Status::OK();
}


arrow::Status ArrowIPCWriter::close () {

    arrow::Status status = writer_->get()->Close();

    if (!status.ok()) {
        // Handle read error
        return status;
    }
    return arrow::Status::OK();
    
}


std::tuple<std::unique_ptr<ApacheArrowWriter>, std::optional<std::string>> WriterFactory::create_writer(const std::string &output_file, const std::vector<std::string> &header) {
    
    if (Nyxus::ends_with_substr(output_file, ".parquet")) {
        
        return {std::make_unique<ParquetWriter>(output_file, header), std::nullopt};

    } else if (Nyxus::ends_with_substr(output_file, ".arrow") || Nyxus::ends_with_substr(output_file, ".feather")) {
        
        return {std::make_unique<ArrowIPCWriter>(output_file, header), std::nullopt};

    } else {

        fs::path path(output_file);

        auto error_msg = [&path](){        
            if (path.has_extension()) 
            {
                return "No writer option for extension \"" + path.extension().string() + "\". Valid options are \".parquet\" or \".arrow\".";
            } else {
                return std::string{"No extension type was provided in the path."};
            }
        }; 

        return {nullptr, error_msg()};
    }
}
#endif