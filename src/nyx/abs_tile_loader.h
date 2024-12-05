#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

template <typename DataType>
class AbstractTileLoader
{
private:
    std::string name_, filePath_;
    size_t numberThreads_;
public:
  /// @brief AbstractTileLoader constructor
  /// @param name AbstractTileLoader name
  /// @param numberThreads Number of AbstractTileLoader used in FastLoader
  /// @param filePath File's path to load
  AbstractTileLoader(std::string_view const &name, size_t numberThreads, std::string filePath)
      : name_(name), numberThreads_(numberThreads), filePath_(filePath) {}

  /// @brief AbstractTileLoader constructor
  /// @param name AbstractTileLoader name
  /// @param filePath File's path to load
  AbstractTileLoader(std::string_view const &name, std::string filePath)
    : name_(name), numberThreads_(1), filePath_(filePath) {}
  /// @brief Default destructor
  virtual ~AbstractTileLoader() = default;

  /// @brief Load a tile from the file at position (indexRowGlobalTile/indexColGlobalTile) for the pyramidal level
  /// "level"
  /// @param tile Tile to load
  /// @param indexRowGlobalTile Tile's row index in the file to load
  /// @param indexColGlobalTile Tile's col index in the file to load
  /// @param indexLayerGlobalTile Tile's layer index in the file to load
  /// @param level Tile's pyramidal level in the file to load
  virtual void loadTileFromFile(std::shared_ptr<std::vector<DataType>> tile,
                                size_t indexRowGlobalTile,
                                size_t indexColGlobalTile,
                                size_t indexLayerGlobalTile,
                                size_t level) = 0;

  /// \brief Getter to full Height
  /// @param level file's level considered
  /// \return Image height
  [[nodiscard]] virtual size_t fullHeight(size_t level) const = 0;

  /// \brief Getter to full Width
  /// @param level file's level considered
  /// \return Image Width
  [[nodiscard]] virtual size_t fullWidth(size_t level) const = 0;

  /// \brief Getter to full Depth (default 1)
  /// @param level file's level considered
  /// \return Image Depth
  [[nodiscard]] virtual size_t fullDepth([[maybe_unused]] size_t level) const {
    return 1;
  }

  /// \brief Getter to the number of channels (default 1)
  /// \return Number of pixel's channels
  [[nodiscard]] virtual size_t numberChannels() const {
    return 1;
  }

  /// \brief Getter to tile Width
  /// @param level tile's level considered
  /// \return Tile Width
  [[nodiscard]] virtual size_t tileWidth(size_t level) const = 0;

  /// \brief Getter to tile Height
  /// @param level tile's level considered
  /// \return Tile Height
  [[nodiscard]] virtual size_t tileHeight(size_t level) const = 0;

  /// \brief Getter to tile Height (default 1)
  /// @param level tile's level considered
  /// \return Tile Height
  [[nodiscard]] virtual size_t tileDepth([[maybe_unused]] size_t level) const {
    return 1;
  }

  /// \brief Get file bits per samples
  /// \return File bits per sample
  [[nodiscard]] virtual short bitsPerSample() const = 0;

  /// \brief Get number of pyramid levels
  /// \return Number of Pyramid levels
  [[nodiscard]] virtual size_t numberPyramidLevels() const = 0;

  /// @brief Number tiles in height accessor for a level
  /// @param level Level asked [default 0]
  /// @return Number tiles in height for a level
  [[nodiscard]] size_t numberTileHeight(size_t level = 0) const {
    return (size_t) std::ceil((double) (fullHeight(level)) / tileHeight(level));
  }
  /// @brief Number tiles in width accessor for a level
  /// @param level Level asked [default 0]
  /// @return Number tiles in width for a level
  [[nodiscard]] size_t numberTileWidth(size_t level = 0) const {
    return (size_t) std::ceil((double) (fullWidth(level)) / tileWidth(level));
  }

  /// @brief Number tiles in depth accessor for a level
  /// @param level Level asked [default 0]
  /// @return Number tiles in depth for a level
  [[nodiscard]] size_t numberTileDepth(size_t level = 0) const {
    return (size_t) std::ceil((double) (fullDepth(level)) / tileDepth(level));
  }
};