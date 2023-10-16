#ifdef USE_ARROW
#include "output_writers.h"

std::shared_ptr<arrow::Table> ApacheArrowWriter::get_arrow_table(const std::string& file_path) {

    if (table_ != nullptr) return table_;

    auto file_extension = fs::path(file_path).extension().u8string();

    if (file_extension == ".parquet") {
        arrow::MemoryPool* pool = arrow::default_memory_pool();


        std::shared_ptr<arrow::io::RandomAccessFile> input;

        input = arrow::io::ReadableFile::Open(file_path).ValueOrDie();
        
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;

        auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);

        if (!status.ok()) {
            // Handle read error
            std::cerr << "Error creating arrow table: " << status.ToString();
            return nullptr;
        }

        // Read entire file as a single Arrow table
        std::shared_ptr<arrow::Table> table;

        status = arrow_reader->ReadTable(&table);

        if (!status.ok()) {
            // Handle read error
            std::cerr << "Error creating arrow table: " << status.ToString();
            return nullptr;
        }

        return table;

    } else if (file_extension == ".arrow") {

        // Create a memory-mapped file for reading.
        std::shared_ptr<arrow::io::ReadableFile> input;
        input = arrow::io::ReadableFile::Open(file_path).ValueOrDie();

        // Create an IPC reader.
        auto ipc_reader = (arrow::ipc::RecordBatchStreamReader::Open(input.get())).ValueOrDie();

        this->table_ = ipc_reader->ToTable().ValueOrDie();

        return table_;

    } else {
        throw std::invalid_argument("Error: file must either be an Arrow or Parquet file.");
    }

}

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


std::shared_ptr<ApacheArrowWriter> WriterFactory::create_writer(const std::string &output_file, const std::vector<std::string> &header) {
    
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

#endif