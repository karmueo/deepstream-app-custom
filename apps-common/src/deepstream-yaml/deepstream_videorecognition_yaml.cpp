/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include "deepstream_videorecognition.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_videorecognition_yaml (NvDsVideoRecognitionConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  // 设置默认值
  config->model_type = 0;  // 0 = multi-frame image classification (default)
  config->enable = FALSE;

  for(YAML::const_iterator itr = configyml["videorecognition"].begin();
     itr != configyml["videorecognition"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "unique-id") {
      config->unique_id = itr->second.as<guint>();
    } else if (paramKey == "gpu-id") {
      config->gpu_id = itr->second.as<guint>();
    } else if (paramKey == "nvbuf-memory-type") {
      config->nvbuf_memory_type = itr->second.as<guint>();
    } else if (paramKey == "batch-size") {
      config->batch_size = itr->second.as<guint>();
    } else if (paramKey == "processing-width") {
      config->processing_width = itr->second.as<guint>();
    } else if (paramKey == "processing-height") {
      config->processing_height = itr->second.as<guint>();
    } else if (paramKey == "model-clip-length") {
      config->model_clip_length = itr->second.as<guint>();
    } else if (paramKey == "num-clips") {
      config->model_num_clips = itr->second.as<guint>();
    } else if (paramKey == "model-type") {
      config->model_type = itr->second.as<guint>();
    } else if (paramKey == "sampling-rate") {
      config->model_sampling_rate = itr->second.as<guint>();
    } else if (paramKey == "trt-engine-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->trt_engine_name = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->trt_engine_name)) {
        g_printerr ("Error: Could not parse trt-engine-file in videorecognition.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else if (paramKey == "labels-file") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1023);
      config->labels_file = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (cfg_file_path, str,
              config->labels_file)) {
        g_printerr ("Error: Could not parse labels-file in videorecognition.\n");
        g_free (str);
        goto done;
      }
      g_free (str);
    } else {
      cout << "Unknown key " << paramKey << " for videorecognition" << endl;
    }
  }

  ret = TRUE;
done:
  if (!ret) {
    cout <<  __func__ << " failed" << endl;
  }
  return ret;
}
