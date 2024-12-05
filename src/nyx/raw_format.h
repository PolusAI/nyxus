#pragma once

#include <string>
#include <string_view>

class RawFormatLoader
{
private:
    std::string name_, filePath_;
    size_t numberThreads_;

public:

    RawFormatLoader (const std::string_view & name, const std::string & filePath)
        : name_(name), filePath_(filePath) {}

    virtual ~RawFormatLoader() = default;

    virtual void loadTileFromFile (
        size_t indexRowGlobalTile,
        size_t indexColGlobalTile,
        size_t indexLayerGlobalTile,
        size_t level) = 0;

    virtual void free_tile() = 0;   // must follow loadTileFromFile()

    [[nodiscard]] virtual uint32_t get_uint32_pixel (size_t idx) const = 0;

    [[nodiscard]] virtual double get_dpequiv_pixel(size_t idx) const = 0;

    [[nodiscard]] virtual size_t fullHeight(size_t level) const = 0;

    [[nodiscard]] virtual size_t fullWidth(size_t level) const = 0;

    [[nodiscard]] virtual size_t fullDepth([[maybe_unused]] size_t level) const {
        return 1;
    }

    [[nodiscard]] virtual size_t numberChannels() const {
        return 1;
    }

    [[nodiscard]] virtual size_t tileWidth(size_t level) const = 0;

    [[nodiscard]] virtual size_t tileHeight(size_t level) const = 0;

    [[nodiscard]] virtual size_t tileDepth([[maybe_unused]] size_t level) const {
        return 1;
    }

    [[nodiscard]] virtual short bitsPerSample() const = 0;

    [[nodiscard]] virtual size_t numberPyramidLevels() const = 0;

    [[nodiscard]] size_t numberTileHeight(size_t level = 0) const {
        return (size_t)std::ceil((double)(fullHeight(level)) / tileHeight(level));
    }

    [[nodiscard]] size_t numberTileWidth(size_t level = 0) const {
        return (size_t)std::ceil((double)(fullWidth(level)) / tileWidth(level));
    }

    [[nodiscard]] size_t numberTileDepth(size_t level = 0) const {
        return (size_t)std::ceil((double)(fullDepth(level)) / tileDepth(level));
    }
};

