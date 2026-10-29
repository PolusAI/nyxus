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
#include "../src/nyx/helpers//fsystem.h"

arrow::Result<std::shared_ptr<arrow::Table>> get_arrow_table(const std::string& file_path) {
    auto file_extension = fs::path(file_path).extension().string();

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

    // 3 leading string columns (intensity, mask, phys_unit) + ROI_label int32 + float64 rest
    const int num_str_cols = 3;
    fields.push_back(arrow::field(header[0], arrow::utf8()));
    fields.push_back(arrow::field(header[1], arrow::utf8()));
    fields.push_back(arrow::field(header[2], arrow::utf8()));
    fields.push_back(arrow::field(header[3], arrow::int32()));

    for (int i = 4; i < header.size(); ++i)
    {
        fields.push_back(arrow::field(header[i], arrow::float64()));
    }

    auto schema = arrow::schema(fields);

    std::vector<std::shared_ptr<arrow::Array>> arrays;

    // de-interleave the flat string buffer (row-major, num_str_cols per row) into one array per column
    for (int c = 0; c < num_str_cols; ++c)
    {
        std::vector<std::string> col(number_of_rows);
        for (int r = 0; r < number_of_rows; ++r)
            col[r] = string_columns[r * num_str_cols + c];
        arrow::StringBuilder sb;
        PARQUET_THROW_NOT_OK(sb.AppendValues(col));
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(sb.Finish(&arr));
        arrays.push_back(arr);
    }

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

void test_arrow_ipc_mechanics() 
{
    auto temp = fs::temp_directory_path() / "nyxus_temp/";

    if (!fs::exists(temp)) {
        auto created = fs::create_directory(temp);
    }

    fs::permissions(temp, fs::perms::all);

    std::string outputPath = temp.string() + "NyxusFeatures.arrow";

    auto arrow_stream = ArrowOutputStream();

    Nyxus::SaveOption saveOption = Nyxus::SaveOption::saveArrowIPC;

    // create arrow writer
    auto [status, msg] = arrow_stream.create_arrow_file(saveOption, outputPath, std::get<0>(features888));
    if (!status) {
        FAIL() << "Error creating Arrow file: " << msg.value() << std::endl;
    }

    // write features
    auto [status1, msg1] = arrow_stream.write_arrow_file(std::get<1>(features888));
    if (!status1) {
        FAIL() << "Error writing Arrow file: " << msg1.value() << std::endl;
    }

    // close arrow file after use
    auto [status2, msg2] = arrow_stream.close_arrow_file();
    if (!status2) {
        FAIL() << "Error closing Arrow file: " << msg2.value() << std::endl;
    }

    auto results_table_result = get_arrow_table(outputPath);
    if (!results_table_result.ok()) {
        FAIL() << "Error reading Arrow file: " << results_table_result.status().ToString() << std::endl;
    }
    auto results_table = results_table_result.ValueOrDie();

    auto& row_data = std::get<1>(features888);
    std::vector<std::string> string_columns;
    std::vector<double> numeric_columns;
    int number_of_rows = row_data.size();

    for (const auto& row : row_data) {
        string_columns.push_back(std::get<0>(row)[0]);
        string_columns.push_back(std::get<0>(row)[1]);
        string_columns.push_back(std::get<0>(row)[2]);    // phys_unit (3rd leading string)
        numeric_columns.push_back(std::get<1>(row));    // ROI label
        numeric_columns.push_back(std::get<FTABLE_TIMEPOS>(row));    // time
        numeric_columns.push_back(std::get<FTABLE_CPOS>(row));    // channel (c_index column)
        numeric_columns.push_back(std::get<FTABLE_PXPOS>(row));    // phys_x
        numeric_columns.push_back(std::get<FTABLE_PYPOS>(row));    // phys_y
        numeric_columns.push_back(std::get<FTABLE_PZPOS>(row));    // phys_z
        for (const auto& data : std::get<FTABLE_FBEGIN>(row)) {
            numeric_columns.push_back(data);
        }
    }

    auto features_table = create_features_table(std::get<0>(features888),
        string_columns,
        numeric_columns,
        number_of_rows);


    ASSERT_TRUE(are_tables_equal(*results_table, *features_table));


    auto is_deleted = fs::remove_all(temp);

    if (!is_deleted) {
        FAIL() << "Error deleting arrow file." << std::endl;
    }
}

void test_arrow_parquet_mechanics() {

    auto temp = fs::temp_directory_path() / "nyxus_temp/";

    if (!fs::exists(temp)) {
        auto created = fs::create_directory(temp);
    }

    fs::permissions(temp, fs::perms::all);

    std::string outputPath = temp.string() + "NyxusFeatures.parquet";

    auto arrow_stream = ArrowOutputStream();

    Nyxus::SaveOption saveOption = Nyxus::SaveOption::saveParquet;

    // create arrow writer
    auto [status, msg] = arrow_stream.create_arrow_file(saveOption, outputPath, std::get<0>(features888));
    if (!status) {
        FAIL() << "Error creating Arrow file: " << msg.value() << std::endl;
    }

    // write features
    auto [status1, msg1] = arrow_stream.write_arrow_file(std::get<1>(features888));
    if (!status1) {
        FAIL() << "Error writing Arrow file: " << msg1.value() << std::endl;
    }

    // close arrow file after use
    auto [status2, msg2] = arrow_stream.close_arrow_file();
    if (!status2) {
        FAIL() << "Error closing Arrow file: " << msg2.value() << std::endl;
    }

    auto results_table_result = get_arrow_table(outputPath);
    if (!results_table_result.ok()) {
        FAIL() << "Error reading Parquet file: " << results_table_result.status().ToString() << std::endl;
    }
    auto results_table = results_table_result.ValueOrDie();

    auto& row_data = std::get<1>(features888);
    std::vector<std::string> string_columns;
    std::vector<double> numeric_columns;
    int number_of_rows = row_data.size();

    for (const auto& row : row_data) {
        string_columns.push_back(std::get<0>(row)[0]);
        string_columns.push_back(std::get<0>(row)[1]);
        string_columns.push_back(std::get<0>(row)[2]);    // phys_unit (3rd leading string)
        numeric_columns.push_back(std::get<1>(row));    // ROI label
        numeric_columns.push_back(std::get<FTABLE_TIMEPOS>(row));    // time
        numeric_columns.push_back(std::get<FTABLE_CPOS>(row));    // channel (c_index column)
        numeric_columns.push_back(std::get<FTABLE_PXPOS>(row));    // phys_x
        numeric_columns.push_back(std::get<FTABLE_PYPOS>(row));    // phys_y
        numeric_columns.push_back(std::get<FTABLE_PZPOS>(row));    // phys_z
        for (const auto& data : std::get<FTABLE_FBEGIN>(row)) {
            numeric_columns.push_back(data);
        }
    }

    auto features_table = create_features_table(std::get<0>(features888),
        string_columns,
        numeric_columns,
        number_of_rows);


    ASSERT_TRUE(are_tables_equal(*results_table, *features_table));


    auto is_deleted = fs::remove_all(temp);

    if (!is_deleted) {
        FAIL() << "Error deleting arrow file." << std::endl;
    }
}

// A target that exists as a directory cannot be opened for writing on any platform, so
// create_arrow_file must report the failure (naming the path) rather than hand back a writer
// whose first write aborts the process. The same stream then writes a file next to it, so
// the refusal is specific to the unopenable path.
void assert_arrow_unopenable_path_refused(Nyxus::SaveOption saveOption, const std::string& ext)
{
    auto temp = fs::temp_directory_path() / ("nyxus_unopenable" + ext);
    fs::remove_all(temp);
    fs::create_directories(temp);

    fs::path blocked = temp / ("NyxusFeatures" + ext);
    fs::create_directory(blocked);

    auto arrow_stream = ArrowOutputStream();

    auto [status, msg] = arrow_stream.create_arrow_file(saveOption, blocked.string(), std::get<0>(features888));
    ASSERT_FALSE(status) << "a writer was created for directory " << blocked.string();
    ASSERT_TRUE(msg.has_value());
    ASSERT_NE(msg->find(blocked.string()), std::string::npos) << "message does not name the path: " << *msg;

    auto [wstatus, wmsg] = arrow_stream.write_arrow_file(std::get<1>(features888));
    ASSERT_FALSE(wstatus);

    std::string writable = (temp / ("writable" + ext)).string();

    auto [status1, msg1] = arrow_stream.create_arrow_file(saveOption, writable, std::get<0>(features888));
    ASSERT_TRUE(status1) << msg1.value_or("");

    auto [status2, msg2] = arrow_stream.write_arrow_file(std::get<1>(features888));
    ASSERT_TRUE(status2) << msg2.value_or("");

    auto [status3, msg3] = arrow_stream.close_arrow_file();
    ASSERT_TRUE(status3) << msg3.value_or("");

    auto table = get_arrow_table(writable);
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    ASSERT_EQ((*table)->num_rows(), (int64_t) std::get<1>(features888).size());

    fs::remove_all(temp);
}

void test_arrow_ipc_unopenable_path_mechanics()
{
    assert_arrow_unopenable_path_refused(Nyxus::SaveOption::saveArrowIPC, ".arrow");
}

void test_arrow_parquet_unopenable_path_mechanics()
{
    assert_arrow_unopenable_path_refused(Nyxus::SaveOption::saveParquet, ".parquet");
}

