#include <iostream>
#include "arrow_output_stream.h"
#include "environment.h"
#include "helpers/helpers.h"


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

namespace Nyxus {

std::string get_arrow_path (const std::string path, const Nyxus::SaveOption& arrow_file_type) {

    bool is_directory = (Nyxus::ends_with_substr(path, "/") ||  path == "" || fs::is_directory(fs::path(path)));

    std::string slash = (path == "" || Nyxus::ends_with_substr(path, "/")) ? "" : "/";

    std::string arrow_file_path = (is_directory) ? path + slash + theEnvironment.nyxus_result_fname : path;

    std::string file_extension = [&arrow_file_type](){
        if (arrow_file_type == Nyxus::SaveOption::saveArrowIPC) {
            return ".arrow";
        } else if (arrow_file_type == Nyxus::SaveOption::saveParquet) {
            return ".parquet";
        } else {return "";}
    }();

    auto current_ext = fs::path(path).extension().u8string();

    if (current_ext == "") {
        arrow_file_path += file_extension;
    } else {
        if (current_ext != file_extension) {
            std::cerr << "Incorrect file extension \"" + current_ext + "\". Using correct extension \"" + file_extension + "\"." << std::endl;

            auto fs_path = fs::path(arrow_file_path);
            arrow_file_path = fs_path.remove_filename().u8string() + fs_path.stem().u8string() + file_extension; // correct the extension
        }
    }
    
    return arrow_file_path;
}

std::tuple<bool, std::optional<std::string>> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                         const std::string& output_path,
                                                         const std::vector<std::string>& header) {

    this->arrow_file_path_ = output_path;

    auto path = fs::path(output_path).remove_filename().u8string();

    if (path != "" && !fs::exists(path)) {
        try {        
            fs::create_directory(path);
        } catch (std::exception& e) {
            return {false, e.what()};
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
}