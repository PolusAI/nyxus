#include "arrow_output_stream.h"
#ifdef USE_ARROW

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {



    auto valid_extension = [&arrow_file_path](){
        if(	arrow_file_path != "" && 
            !fs::is_directory(arrow_file_path)){
                if( auto ext = fs::path(arrow_file_path).extension(); 
                    ext == ".arrow" || ext == ".feather" || ext == ".arrow"){
                        return true;
                } else {
                    return false;
                } 
            }
        return false;
    }(); 

    if (valid_extension) {
        arrow_file_path_ = arrow_file_path;
    } else {
        std::string extension = (arrow_file_type == Nyxus::SaveOption::saveParquet) ? "parquet" : "arrow";
        if (arrow_file_path == "") {
            arrow_file_path_ = "NyxusFeatures." + extension;
        } else if (fs::is_directory(arrow_file_path)) {
            arrow_file_path_ = arrow_file_path + "/NyxusFeatures." + extension;
        } else {
            arrow_file_path_ = fs::path(arrow_file_path).replace_extension(extension);
        }
    }

    std::optional<std::string> error_msg;
    std::tie(writer_, error_msg) = WriterFactory::create_writer(arrow_file_path_, header);
    if (writer_) {
        return {true, std::nullopt};
    } else {
        return {false, error_msg};
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
            return {true, std::nullopt};
        }
        else {
            return {false, status.ToString()};
        }
    }
    return {false, "Arrow Writer is not initialized."};
}
std::tuple<bool, std::optional<std::string>> ArrowOutputStream::close_arrow_file (){
    if (writer_){
        auto status = writer_->close();
        if (status.ok()) {
            return {true, std::nullopt};
        }
        else {
            return {false, status.ToString()};
        }
    }
    return {false, "Arrow Writer is not initialized."};
}

#else 

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& arrow_file_path,
                                                         const std::vector<std::string>& header) {
    
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;

    return {false, "Apache Arrow functionality is not available."};
}

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::write_arrow_file (const std::vector<std::tuple<std::vector<std::string>, int, std::vector<double>>>& features){
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    return {false, "Apache Arrow functionality is not available."};
}
std::tuple<bool, std::optional<std::string>> ArrowOutputStream::close_arrow_file (){
    std::cerr << "Apache Arrow functionality is not available. Please install Nyxus with Arrow enabled to use this functionality." << std::endl;
    return {false, "Apache Arrow functionality is not available."};
}


#endif