/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

/*!
 * @brief Image data container (image or video frame stack)
 *
 * Wraps a uint8 RGB tensor with the unified 4D layout `[frames, height, width, channels]`. Single still
 * images are represented as `frames == 1`; video frame stacks have `frames > 1`. Channels must be 3.
 */
class ImageData
{
public:
    std::shared_ptr<rt::Tensor> buffer; //!< Image data buffer (4D [T, H, W, C], UINT8, channels == 3)
    int64_t width{0};                   //!< Image width
    int64_t height{0};                  //!< Image height
    int64_t channels{0};                //!< Number of channels (e.g., 3 for RGB)
    int64_t frames{1};                  //!< Number of frames (T). 1 for static images, >1 for video.
    double fps{1.0}; //!< Video sample fps (used to compute MRoPE timestamps); ignored when frames == 1.

    /*!
     * @brief Default constructor (creates uninitialized ImageData)
     */
    ImageData() noexcept = default;

    /*!
     * @brief Construct image data
     * @param data Image tensor with shape [T, H, W, C]. Single-frame still images use T=1.
     * @throws std::runtime_error if tensor content not UINT8, tensor shape not 4D, or number of channels not 3
     */
    ImageData(rt::Tensor&& data);

    //! @brief Get raw image data pointer
    //! @return Pointer to image data
    unsigned char* data() const noexcept;

    //! @brief Bytes occupied by a single frame (height * width * channels), assuming UINT8 storage.
    //! @return Byte count per frame.
    int64_t bytesPerFrame() const noexcept
    {
        return height * width * channels;
    }
};

/*!
 * @brief Load image from file
 * @param path Path to image file
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from file, or memory allocation fails
 */
ImageData loadImageFromFile(std::string const& path);

/*!
 * @brief Load image from memory
 * @param data Pointer to image data in memory
 * @param size Size of image data in bytes
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from memory, or memory allocation fails
 */
ImageData loadImageFromMemory(unsigned char const* data, size_t size);

/*!
 * @brief Load a video by stacking a list of identically-sized image files into a single 4D `[T, H, W, C]` tensor
 * @param framePaths One file path per video frame (in temporal order)
 * @param fps Source frame rate used by the runner to compute MRoPE timestamps
 * @return Loaded video as a single ImageData with `frames == framePaths.size()`
 * @throws std::runtime_error if framePaths is empty, any frame fails to load, frame sizes mismatch, or memory
 * allocation fails
 */
ImageData loadVideoFromFrames(std::vector<std::string> const& framePaths, double fps = 1.0);

//! @brief Interpolation filter for :func:`resizeImage`.
enum class InterpolationMode
{
    kLINEAR,  //!< Bilinear
    kBICUBIC, //!< Catmull-Rom cubic
};

/*!
 * @brief Resize each frame of an image/video stack into a pre-allocated buffer
 * @param image Source image (4D `[T, H, W, C]`)
 * @param resizedImage Output buffer (reshaped to target dimensions)
 * @param newWidth Target width
 * @param newHeight Target height
 * @param mode Interpolation filter
 * @throws std::runtime_error if image buffer cannot be reshaped
 *
 * TODO(perf): per-frame CPU stbir is slow for video; consider GPU resize.
 */
void resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
