/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include "deepstream_cuav_control.h"
#include <cstring>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

gboolean
parse_cuav_control_yaml(NvDsCuavControlConfig *config, gchar *cfg_file_path)
{
  gboolean ret = FALSE;
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  if (!configyml["cuav-control"]) {
    return TRUE;
  }

  config->enable = FALSE;
  config->multicast_ip = NULL;
  config->port = 18003;
  config->iface = NULL;
  config->ttl = 1;
  config->compat_cmd_wrapper = FALSE;
  config->debug = FALSE;
  config->tx_sys_id = 999;
  config->tx_dev_type = 1;
  config->tx_dev_id = 999;
  config->tx_subdev_id = 999;
  config->rx_sys_id = 999;
  config->rx_dev_type = 1;
  config->rx_dev_id = 999;
  config->rx_subdev_id = 999;
  config->send_test_on_startup = FALSE;

  for (YAML::const_iterator itr = configyml["cuav-control"].begin();
       itr != configyml["cuav-control"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enable") {
      config->enable = itr->second.as<gboolean>();
    } else if (paramKey == "multicast-ip") {
      std::string temp = itr->second.as<std::string>();
      config->multicast_ip = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->multicast_ip, temp.c_str(), 1023);
      config->multicast_ip[1023] = '\0';
    } else if (paramKey == "port") {
      config->port = itr->second.as<guint>();
    } else if (paramKey == "interface") {
      std::string temp = itr->second.as<std::string>();
      config->iface = (char *)malloc(sizeof(char) * 1024);
      std::strncpy(config->iface, temp.c_str(), 1023);
      config->iface[1023] = '\0';
    } else if (paramKey == "ttl") {
      config->ttl = itr->second.as<guint>();
    } else if (paramKey == "compat-cmd-wrapper") {
      config->compat_cmd_wrapper = itr->second.as<gboolean>();
    } else if (paramKey == "debug") {
      config->debug = itr->second.as<gboolean>();
    } else if (paramKey == "tx-sys-id") {
      config->tx_sys_id = itr->second.as<guint>();
    } else if (paramKey == "tx-dev-type") {
      config->tx_dev_type = itr->second.as<guint>();
    } else if (paramKey == "tx-dev-id") {
      config->tx_dev_id = itr->second.as<guint>();
    } else if (paramKey == "tx-subdev-id") {
      config->tx_subdev_id = itr->second.as<guint>();
    } else if (paramKey == "rx-sys-id") {
      config->rx_sys_id = itr->second.as<guint>();
    } else if (paramKey == "rx-dev-type") {
      config->rx_dev_type = itr->second.as<guint>();
    } else if (paramKey == "rx-dev-id") {
      config->rx_dev_id = itr->second.as<guint>();
    } else if (paramKey == "rx-subdev-id") {
      config->rx_subdev_id = itr->second.as<guint>();
    } else if (paramKey == "send-test-on-startup") {
      config->send_test_on_startup = itr->second.as<gboolean>();
    } else {
      cout << "Unknown key " << paramKey << " for cuav-control" << endl;
    }
  }

  ret = TRUE;
  if (!ret) {
    cout << __func__ << " failed" << endl;
  }
  return ret;
}
