#include "arrow_output_stream.h"
#ifdef USE_ARROW

bool ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {



    std::string extension = (arrow_file_type == Nyxus::SaveOption::saveParquet) ? ".parquet" : ".arrow";

    if (arrow_file_path == "") {
        arrow_file_path_ = "NyxusFeatures" + extension;
    } else if (fs::is_directory(arrow_file_path)) {
        arrow_file_path_ = arrow_file_path + "/NyxusFeatures" + extension;
    } else {
        arrow_file_path_ = arrow_file_path;
    }
    
    std::optional<std::string> error_msg;
    std::tie(writer_, error_msg) = WriterFactory::create_writer(arrow_file_path_, header);
    if (writer_) {
        return true;
    } else {
        std::cout << error_msg.value() << std::endl;
        return false;
    }
}


std::shared_ptr<arrow::Table> ArrowOutputStream::get_arrow_table(const std::string& file_path) {

    if (this->arrow_table_ != nullptr) return this->arrow_table_;
                                                    
    this->arrow_table_ = writer_->get_arrow_table(file_path);

    return this->arrow_table_;
}

std::string ArrowOutputStream::get_arrow_path() {
    return arrow_file_path_;
}

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::write_arrow_file (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features){
    if (writer_){
        auto status = writer_->write(features);
        if (status.ok()) {
            return std::make_tuple(true, std::nullopt);
        }
        else {
            return std::make_tuple(false, status.ToString());
        }
    }
    return std::make_tuple(false, "Arrow Writer is not initialized.");
}
std::tuple<bool, std::optional<std::string>> ArrowOutputStream::close_arrow_file (){
    if (writer_){
        auto status = writer_->close();
        if (status.ok()) {
            return std::make_tuple(true, std::nullopt);
        }
        else {
            return std::make_tuple(false, status.ToString());
        }
    }
    return std::make_tuple(false, "Arrow Writer is not initialized.");
}

#else 

bool ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {
    
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;

    return false;
}


bool ArrowOutputStream::get_arrow_table(const std::string& file_path) {

    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;

    return false;
}

std::string ArrowOutputStream::get_arrow_path() {

    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    
    return "";
}

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::write_arrow_file (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features){
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    return std::make_tuple(false, "Apache Arrow functionality is not available.")
}
std::tuple<bool, std::optional<std::string>> ArrowOutputStream::close_arrow_file (){
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    return std::make_tuple(false, "Apache Arrow functionality is not available.")
}


#endif