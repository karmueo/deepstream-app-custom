/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "deepstream_udpjsonmeta.h"
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

/**
 * @brief 解析 UDP JSON 元数据插件的 YAML 配置。
 *
 * @param config UDP JSON 元数据配置结构体指针。
 * @param cfg_file_path 配置文件路径。
 * @return 成功返回 TRUE。
 */
gboolean
parse_udpjsonmeta_yaml (NvDsUdpJsonMetaConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE; /* 返回值 */
  YAML::Node configyml = YAML::LoadFile(cfg_file_path); /* YAML 节点 */

  if (!configyml["udpjsonmeta"]) {
    return TRUE;
  }

  config->enable = FALSE;
  config->multicast_ip = NULL;
  config->port = 0;
  config->iface = NULL;
  config->recv_buf_size = 0;
  config->json_key = NULL;
  config->object_id_key = NULL;
  config->source_id_key = NULL;
  config->cache_ttl_ms = 0;
  config->max_cache_size = 0;

  for (YAML::const_iterator itr = configyml["udpjsonmeta"].begin();
       itr != configyml["udpjsonmeta"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>(); /* 参数名 */
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "multicast-ip") {
      std::string temp = itr->second.as<std::string>(); /* 组播地址 */
      config->multicast_ip = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->multicast_ip, temp.c_str(), 1023);
    } else if (paramKey == "port") {
      config->port = itr->second.as<guint>();
    } else if (paramKey == "interface") {
      std::string temp = itr->second.as<std::string>(); /* 网卡名 */
      config->iface = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->iface, temp.c_str(), 1023);
    } else if (paramKey == "recv-buf-size") {
      config->recv_buf_size = itr->second.as<guint>();
    } else if (paramKey == "json-key") {
      std::string temp = itr->second.as<std::string>(); /* JSON 值键 */
      config->json_key = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->json_key, temp.c_str(), 1023);
    } else if (paramKey == "object-id-key") {
      std::string temp = itr->second.as<std::string>(); /* 目标ID键 */
      config->object_id_key = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->object_id_key, temp.c_str(), 1023);
    } else if (paramKey == "source-id-key") {
      std::string temp = itr->second.as<std::string>(); /* 源ID键 */
      config->source_id_key = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->source_id_key, temp.c_str(), 1023);
    } else if (paramKey == "cache-ttl-ms") {
      config->cache_ttl_ms = itr->second.as<guint>();
    } else if (paramKey == "max-cache-size") {
      config->max_cache_size = itr->second.as<guint>();
    } else {
      cout << "Unknown key " << paramKey << " for udpjsonmeta" << endl;
    }
  }

  ret = TRUE;
  if (!ret) {
    cout << __func__ << " failed" << endl;
  }
  return ret;
}
