/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef _NVGSTDS_VIDEORECOGNITION_H_
#define _NVGSTDS_VIDEORECOGNITION_H_

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        // Create a bin for the element only if enabled
        gboolean enable;
        guint unique_id;
        guint gpu_id;
        guint batch_size;
        // For nvvidconv
        guint nvbuf_memory_type;
        guint processing_width;
        guint processing_height;
        guint model_clip_length;
        guint model_num_clips;
    // 0 = multi-frame image classification (default), 1 = video recognition (temporal/clip based)
    guint model_type;
        gchar *trt_engine_name; // TensorRT engine name
    } NvDsVideoRecognitionConfig;

    // Struct to store references to the bin and elements
    typedef struct
    {
        GstElement *bin;
        GstElement *queue;
        GstElement *pre_conv;
        GstElement *cap_filter;
        GstElement *elem_dsvideorecognition;
    } NvDsVideoRecognitionBin;

    // Function to create the bin and set properties
    gboolean
    create_dsvideorecognition_bin(NvDsVideoRecognitionConfig *config, NvDsVideoRecognitionBin *bin);

#ifdef __cplusplus
}
#endif

#endif /* _NVGSTDS_VIDEORECOGNITION_H_ */
