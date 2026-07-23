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

#include "runtime/imageUtils.h"
#include "common/checkMacros.h"
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

ImageData::ImageData(rt::Tensor&& data)
{
    check::check(data.getDataType() == nvinfer1::DataType::kUINT8, "Image data must be UINT8");
    check::check(data.getShape().getNumDims() == 4, "Image data must be 4D [T, H, W, C]");
    check::check(data.getShape()[3] == 3, "Image data must have 3 channels");

    // [frames, height, width, channels]
    frames = data.getShape()[0];
    height = data.getShape()[1];
    width = data.getShape()[2];
    channels = data.getShape()[3];
    check::check(frames > 0, "Image frame sequence must have frames > 0");
    buffer = std::make_shared<rt::Tensor>(std::move(data));
}

unsigned char* ImageData::data() const noexcept
{
    return buffer ? buffer->dataPointer<unsigned char>() : nullptr;
}

ImageData loadImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load(path.c_str(), &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({1, height, width, desiredChannels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadImageFromFile::imgTensor");
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * desiredChannels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData loadImageFromMemory(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image from memory: " + std::string(stbi_failure_reason()));

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({1, height, width, desiredChannels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadImageFromMemory::imgTensor");
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * desiredChannels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

ImageData loadVideoFromFrames(std::vector<std::string> const& framePaths, double const fps)
{
    ELLM_CHECK(!framePaths.empty(), "loadVideoFromFrames: framePaths is empty");
    ELLM_CHECK(fps > 0.0, "loadVideoFromFrames: fps must be positive, got " + std::to_string(fps));

    // Load the first frame to determine the common (H, W, C).
    ImageData firstFrame = loadImageFromFile(framePaths[0]);
    int64_t const T = static_cast<int64_t>(framePaths.size());
    int64_t const H = firstFrame.height;
    int64_t const W = firstFrame.width;
    int64_t const C = firstFrame.channels;
    int64_t const frameBytes = H * W * C;

    rt::Tensor stacked{};
    try
    {
        stacked = rt::Tensor(
            {T, H, W, C}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "imageUtils::loadVideoFromFrames::stacked");
    }
    catch (std::exception const& e)
    {
        throw std::runtime_error("Failed to allocate space for video tensor: " + std::string(e.what()));
    }
    unsigned char* dst = stacked.dataPointer<unsigned char>();
    std::memcpy(dst, firstFrame.data(), frameBytes);

    for (int64_t t = 1; t < T; ++t)
    {
        ImageData frame = loadImageFromFile(framePaths[t]);
        ELLM_CHECK(frame.height == H && frame.width == W && frame.channels == C,
            "loadVideoFromFrames: frame " + std::to_string(t) + " has shape (" + std::to_string(frame.height) + ", "
                + std::to_string(frame.width) + ", " + std::to_string(frame.channels) + ") but expected ("
                + std::to_string(H) + ", " + std::to_string(W) + ", " + std::to_string(C) + ")");
        std::memcpy(dst + t * frameBytes, frame.data(), frameBytes);
    }

    ImageData video(std::move(stacked));
    video.fps = fps;
    return video;
}

void resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode)
{
    // Reshape pre-allocated buffer to target [T, H, W, C] (always 4D).
    bool const success = resizedImage.buffer->reshape({image.frames, newHeight, newWidth, image.channels});
    ELLM_CHECK(success, "Failed to reshape resized image buffer");
    resizedImage.frames = image.frames;
    resizedImage.height = newHeight;
    resizedImage.width = newWidth;
    resizedImage.channels = image.channels;
    resizedImage.fps = image.fps; // preserve sample fps (drives Qwen2.5-VL video MRoPE time interval)

    // Resize the image(s) into the pre-allocated buffer. stbir is invoked once per frame.
    constexpr int32_t kINPUT_STRIDE_BYTES{0};
    constexpr int32_t kOUTPUT_STRIDE_BYTES{0};
    int64_t const srcFrameBytes = image.bytesPerFrame();
    int64_t const dstFrameBytes = newHeight * newWidth * image.channels;
    for (int64_t t = 0; t < image.frames; ++t)
    {
        unsigned char const* src = image.data() + t * srcFrameBytes;
        unsigned char* dst = resizedImage.data() + t * dstFrameBytes;
        if (mode == InterpolationMode::kBICUBIC)
        {
            stbir_resize(src, image.width, image.height, kINPUT_STRIDE_BYTES, dst, newWidth, newHeight,
                kOUTPUT_STRIDE_BYTES, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
        }
        else
        {
            stbir_resize_uint8_linear(src, image.width, image.height, kINPUT_STRIDE_BYTES, dst, newWidth, newHeight,
                kOUTPUT_STRIDE_BYTES, STBIR_RGB);
        }
    }
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm
