#include "arrow_output_stream.h"

#ifdef USE_ARROW

std::shared_ptr<ApacheArrowWriter> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {

    if(arrow_file_path != "" && !fs::is_directory(arrow_file_path) && !(Nyxus::ends_with_substr(arrow_file_path, ".arrow") || Nyxus::ends_with_substr(arrow_file_path, ".feather") || Nyxus::ends_with_substr(arrow_file_path, ".parquet"))) {
        throw std::invalid_argument("The arrow file path must end in \".arrow\"");
    }

    if (arrow_file_type != Nyxus::SaveOption::saveArrowIPC && arrow_file_type != Nyxus::SaveOption::saveParquet) {
        throw std::invalid_argument("The valid save options are Nyxus::SaveOption::saveArrowIPC or Nyxus::SaveOption::saveParquet.");
    }

    std::string extension = (arrow_file_type == Nyxus::SaveOption::saveParquet) ? ".parquet" : ".arrow";

    if (arrow_file_path == "") {
        arrow_file_path_ = "NyxusFeatures" + extension;
    } else {
        arrow_file_path_ = arrow_file_path;
    }

    if (fs::is_directory(arrow_file_path)) {
        arrow_file_path_ += "/NyxusFeatures" + extension;
    }

    writer_ = WriterFactory::create_writer(arrow_file_path_, header);

    return writer_;
}


std::shared_ptr<arrow::Table> ArrowOutputStream::get_arrow_table(const std::string& file_path) {

    if (this->arrow_table_ != nullptr) return this->arrow_table_;
                                                    
    this->arrow_table_ = writer_->get_arrow_table(file_path);

    return this->arrow_table_;
}

std::string ArrowOutputStream::get_arrow_path() {
    return arrow_file_path_;
}

#else 

std::shared_ptr<ApacheArrowWriter> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {
    
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;

    return nullptr;
}


std::shared_ptr<arrow::Table> ArrowOutputStream::get_arrow_table(const std::string& file_path) {

    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;

    return nullptr;
}

std::string ArrowOutputStream::get_arrow_path() {

    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    
    return "";
}

#endif