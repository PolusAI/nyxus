#pragma once

// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.
//
// Created by Bardakoff, Alexandre (IntlAssoc) on 10/28/20.
//

#ifndef INC_3DFASTLOADER_VIRTUAL_FILE_TILE_CHANNEL_LOADER_H
#define INC_3DFASTLOADER_VIRTUAL_FILE_TILE_CHANNEL_LOADER_H

#include <fast_loader/fast_loader.h> 

/// @brief Encapsulates a FastLoader-based browser of a multi-tile TIFF image
class VirtualFileTileChannelLoader : public fl::AbstractTileLoader<fl::DefaultView<int>> {
	uint16_t const
		fileHeight_ = 0,
		fileWidth_ = 0,
		fileDepth_ = 0,
		tileHeight_ = 0,
		tileWidth_ = 0,
		tileDepth_ = 0,
		numberChannels_ = 0;

	std::vector<int>
		file_{};

public:
	VirtualFileTileChannelLoader(size_t numberThreads,
		uint16_t const fileHeight,
		uint16_t const fileWidth,
		uint16_t const fileDepth,
		uint16_t const tileHeight,
		uint16_t const tileWidth,
		uint16_t const tileDepth,
		uint16_t const numberChannels) : AbstractTileLoader("3DTileLoader", numberThreads, "filePath"),
		fileHeight_(fileHeight),
		fileWidth_(fileWidth),
		fileDepth_(fileDepth),
		tileHeight_(tileHeight),
		tileWidth_(tileWidth),
		tileDepth_(tileDepth),
		numberChannels_(numberChannels) {
		file_ = std::vector<int>(fileDepth_ * fileWidth_ * fileHeight_ * numberChannels_, 0);
		for (auto layer = 0; layer < fileDepth_; ++layer) {
			for (auto row = 0; row < fileHeight_; ++row) {
				for (auto col = 0; col < fileWidth_; ++col) {
					for (auto channel = 0; channel < numberChannels_; ++channel) {
						file_.at((layer * fileHeight_ * fileWidth_ + row * fileWidth_ + col) * numberChannels + channel)
							= (layer + 1) * 100 + (row + 1) * 10 + col + 1;
					}
				}
			}
		}
	}

	~VirtualFileTileChannelLoader() override = default;

	void loadTileFromFile(std::shared_ptr<std::vector<int>> ptr,
		uint32_t indexRowGlobalTile,
		uint32_t indexColGlobalTile,
		uint32_t indexLayerGlobalTile,
		[[maybe_unused]] uint32_t level) override {
		size_t
			startingLayer = indexLayerGlobalTile * tileDepth_,
			endLayer = std::min((size_t)((indexLayerGlobalTile + 1) * tileDepth_), (size_t)fileDepth_),
			startingRow = indexRowGlobalTile * tileHeight_,
			endRow = std::min((size_t)((indexRowGlobalTile + 1) * tileHeight_), (size_t)fileHeight_),
			startingCol = indexColGlobalTile * tileWidth_,
			endCol = std::min((size_t)((indexColGlobalTile + 1) * tileWidth_), (size_t)fileWidth_),
			width = endCol - startingCol;


		for (size_t layer = startingLayer; layer < endLayer; ++layer) {
			for (size_t row = startingRow; row < endRow; ++row) {

				std::copy_n(
					file_.cbegin()
					+ (layer * fileHeight_ * fileWidth_ + row * fileWidth_ + startingCol)
					* numberChannels_,
					width * numberChannels_,
					ptr->begin()
					+ ((layer - startingLayer) * tileHeight_ * tileWidth_ + (row - startingRow) * tileWidth_)
					* numberChannels_
				);
			}
		}
	}

	[[nodiscard]] std::shared_ptr<AbstractTileLoader> copyTileLoader() override {
		return std::make_shared<VirtualFileTileChannelLoader>(
			this->numberThreads(),
			fileHeight_, fileWidth_, fileDepth_,
			tileHeight_, tileWidth_, tileDepth_,
			numberChannels_
			);
	}

	[[nodiscard]] uint32_t fullHeight([[maybe_unused]] uint32_t level) const override { return fileHeight_; }
	[[nodiscard]] uint32_t fullWidth([[maybe_unused]] uint32_t level) const override { return fileWidth_; }
	[[nodiscard]] uint32_t fullDepth([[maybe_unused]] uint32_t level) const override { return fileDepth_; }

	[[nodiscard]] uint32_t tileWidth([[maybe_unused]] uint32_t level) const override { return tileWidth_; }
	[[nodiscard]] uint32_t tileHeight([[maybe_unused]] uint32_t level) const override { return tileHeight_; }
	[[nodiscard]] uint32_t tileDepth([[maybe_unused]] uint32_t level) const override { return tileDepth_; }
	[[nodiscard]] uint32_t numberChannels() const override { return numberChannels_; }

	[[nodiscard]] short bitsPerSample() const override { return sizeof(int); }
	[[nodiscard]] uint32_t numberPyramidLevels() const override { return 1; }
};

#endif //INC_3DFASTLOADER_VIRTUAL_FILE_TILE_CHANNEL_LOADER_H


