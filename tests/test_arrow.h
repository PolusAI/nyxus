#pragma once

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#include <arrow/ipc/api.h>
#include <arrow/result.h>
#include <arrow/ipc/reader.h>

#include <arrow/csv/api.h>

#include "test_data.h"

#include "../src/nyx/arrow_output_stream.h"
#include "../src/nyx/output_writers.h"
#include "../src/nyx/save_option.h"

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

arrow::Result<std::shared_ptr<arrow::Table>> get_arrow_table(const std::string& file_path) {
    auto file_extension = fs::path(file_path).extension().u8string();

    if (file_extension == ".parquet") {
        arrow::MemoryPool* pool = arrow::default_memory_pool();

        ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(file_path));

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        ARROW_ASSIGN_OR_RAISE(arrow_reader, parquet::arrow::OpenFile(input, pool));

        std::shared_ptr<arrow::Table> table;
        ARROW_RETURN_NOT_OK(arrow_reader->ReadTable(&table));

        return table;

    } else if (file_extension == ".arrow") {
        ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(file_path));

        std::shared_ptr<arrow::ipc::RecordBatchFileReader> reader;
        ARROW_ASSIGN_OR_RAISE(reader, arrow::ipc::RecordBatchFileReader::Open(input.get()));

        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int i = 0; i < reader->num_record_batches(); ++i) {
            ARROW_ASSIGN_OR_RAISE(auto batch, reader->ReadRecordBatch(i));
            batches.push_back(batch);
        }

        ARROW_ASSIGN_OR_RAISE(auto table, arrow::Table::FromRecordBatches(batches));
        return table;

    } else {
        return arrow::Status::Invalid("Error: file must either be an Arrow or Parquet file.");
    }
}

std::shared_ptr<arrow::Table> create_features_table(const std::vector<std::string> &header,
                                                    const std::vector<std::string> &string_columns,
                                                    const std::vector<double> &numeric_columns,
                                                    int number_of_rows){

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

bool are_tables_equal(const arrow::Table& table1, const arrow::Table& table2) {
    // Compare schemas
    if (!table1.schema()->Equals(*table2.schema())) {
        std::cout << "Schema are not equal" << std::endl;

        std::cout << "table1 schema: " << table1.schema()->ToString() << std::endl;
        std::cout << std::endl;
        std::cout << "--------------------------" << std::endl;
        std::cout << "table2 schema: " << table2.schema()->ToString() << std::endl;
        return false;
    }

    // Compare each column
    for (int i = 0; i < table1.num_columns(); ++i) {
        auto column1 = table1.column(i);
        auto column2 = table2.column(i);
        

        // Compare column data
        if (!column1->ApproxEquals(*column2)) {
            std::cout << "-------------------columns:-------------------" << std::endl;
            std::cout << "1: " << column1->ToString() << std::endl;
            std::cout << "2: " << column2->ToString() << std::endl;
            std::cout << "--------------------------------------" << std::endl;
            return false;
        }
    }

    return true;
}


void test_arrow() {

    auto temp = fs::temp_directory_path()/"nyxus_temp/";

    if(!fs::exists(temp)) {
        auto created = fs::create_directory(temp);
    }

    fs::permissions(temp, fs::perms::all);

    std::string outputPath = temp.u8string() + "NyxusFeatures.arrow";

    auto arrow_stream = ArrowOutputStream();

    Nyxus::SaveOption saveOption = Nyxus::SaveOption::saveArrowIPC;

    // create arrow writer
    auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(saveOption, outputPath, std::get<0>(features));
    if (!status) {
        FAIL() << "Error creating Arrow file: " << msg.value() << std::endl;
    }

    // write features
    auto [status1, msg1] = theEnvironment.arrow_stream.write_arrow_file(std::get<1>(features));
    if (!status1) {
        FAIL() << "Error writing Arrow file: " << msg1.value() << std::endl;
    }

    // close arrow file after use
    auto [status2, msg2] = theEnvironment.arrow_stream.close_arrow_file();
    if (!status2) {
        FAIL() << "Error closing Arrow file: " << msg2.value() << std::endl;
    }

    auto results_table_result = get_arrow_table(outputPath);
    if (!results_table_result.ok()) {
        FAIL() << "Error reading Arrow file: " << results_table_result.status().ToString() << std::endl;
    }
    auto results_table = results_table_result.ValueOrDie();

    auto& row_data = std::get<1>(features);
    std::vector<std::string> string_columns;
    std::vector<double> numeric_columns;
    int number_of_rows = row_data.size();

    for(const auto& row: row_data){
        string_columns.push_back(std::get<0>(row)[0]);
        string_columns.push_back(std::get<0>(row)[1]);
        numeric_columns.push_back(std::get<1>(row));
        for (const auto& data: std::get<2>(row)) {
            numeric_columns.push_back(data);
        }
    }

    auto features_table = create_features_table(std::get<0>(features),
                                                string_columns,
                                                numeric_columns,
                                                number_of_rows);


    ASSERT_TRUE(are_tables_equal(*results_table, *features_table));


    auto is_deleted = fs::remove_all(temp);

    if(!is_deleted) {
        FAIL() << "Error deleting arrow file." << std::endl;
    }
}

void test_parquet() {

    auto temp = fs::temp_directory_path()/"nyxus_temp/";

    if(!fs::exists(temp)) {
        auto created = fs::create_directory(temp);
    }

    fs::permissions(temp, fs::perms::all);
    
    std::string outputPath = temp.u8string() + "NyxusFeatures.parquet";

    auto arrow_stream = ArrowOutputStream();

    Nyxus::SaveOption saveOption = Nyxus::SaveOption::saveParquet;

    // create arrow writer
    auto [status, msg] = theEnvironment.arrow_stream.create_arrow_file(saveOption, outputPath, std::get<0>(features));
    if (!status) {
        FAIL() << "Error creating Arrow file: " << msg.value() << std::endl;
    }

    // write features
    auto [status1, msg1] = theEnvironment.arrow_stream.write_arrow_file(std::get<1>(features));
    if (!status1) {
        FAIL() << "Error writing Arrow file: " << msg1.value() << std::endl;
    }

    // close arrow file after use
    auto [status2, msg2] = theEnvironment.arrow_stream.close_arrow_file();
    if (!status2) {
        FAIL() << "Error closing Arrow file: " << msg2.value() << std::endl;
    }

    auto results_table_result = get_arrow_table(outputPath);
    if (!results_table_result.ok()) {
        FAIL() << "Error reading Parquet file: " << results_table_result.status().ToString() << std::endl;
    }
    auto results_table = results_table_result.ValueOrDie();

    auto& row_data = std::get<1>(features);
    std::vector<std::string> string_columns;
    std::vector<double> numeric_columns;
    int number_of_rows = row_data.size();

    for(const auto& row: row_data){
        string_columns.push_back(std::get<0>(row)[0]);
        string_columns.push_back(std::get<0>(row)[1]);
        numeric_columns.push_back(std::get<1>(row));
        for (const auto& data: std::get<2>(row)) {
            numeric_columns.push_back(data);
        }
    }

    auto features_table = create_features_table(std::get<0>(features),
                                                string_columns,
                                                numeric_columns,
                                                number_of_rows);


    ASSERT_TRUE(are_tables_equal(*results_table, *features_table));


    auto is_deleted = fs::remove_all(temp);

    if(!is_deleted) {
        FAIL() << "Error deleting arrow file." << std::endl;
    }
}