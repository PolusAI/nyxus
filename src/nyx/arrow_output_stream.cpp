#include <iostream>
#include "arrow_output_stream.h"

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

#ifdef USE_ARROW
    std::tuple<bool, std::optional<std::string>> ArrowOutputStream::create_arrow_file(const Nyxus::SaveOption& arrow_file_type,
                                                            const std::string& output_path,
                                                            const std::vector<std::string>& header) {

        if (output_path == ""){
            return {false, "No path provided for Arrow file."};
        }

        this->arrow_file_path_ = output_path;
        if (auto arrow_path = fs::path(output_path); !fs::is_directory(arrow_path.parent_path())){
            try {        
                fs::create_directory(arrow_path.parent_path());
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
};