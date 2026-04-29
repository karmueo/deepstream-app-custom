/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include "deepstream_cuav_control.h"
#include <cmath>
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
  config->debug = FALSE;
  config->print_upstream_state = FALSE;
  config->tx_sys_id = 999;
  config->tx_dev_type = 1;
  config->tx_dev_id = 999;
  config->tx_subdev_id = 999;
  config->rx_sys_id = 999;
  config->rx_dev_type = 1;
  config->rx_dev_id = 999;
  config->rx_subdev_id = 999;
  config->send_test_on_startup = FALSE;
  config->auto_track_enable = FALSE;
  config->control_source_id = 0;
  config->control_period_ms = 100;
  config->target_lost_hold_ms = 1000;
  config->tracking_history_size = 5;
  config->center_deadband_x = 0.03;
  config->center_deadband_y = 0.03;
  config->servo_kp_x = 1.5;
  config->servo_kp_y = 1.5;
  config->servo_kv_x = 0.35;
  config->servo_kv_y = 0.35;
  config->servo_dir_x = 1;
  config->servo_dir_y = -1;
  config->servo_max_step_h = 1.5;
  config->servo_max_step_v = 1.0;
  config->servo_focal_adaptive_enable = TRUE;
  config->servo_focal_max_step_scale_min = 0.25;
  config->servo_focal_speed_scale_min = 0.50;
  config->servo_min_speed = 10;
  config->servo_max_speed = 60;
  config->zoom_target_ratio_min = 0.20;
  config->zoom_target_ratio_max = 0.35;
  config->zoom_deadband = 0.02;
  config->visible_focal_hold_ms = 300;
  config->visible_light_control_enable = TRUE;
  config->servo_dev_id = 2;
  config->pt_focal_min = 134.0;
  config->pt_focal_max = 16298.0;
  config->servo_effect_threshold_h = 0.5;
  config->servo_effect_threshold_v = 0.3;
  config->state_stale_timeout_ms = 2000;
  config->corner_zoom_cycle_enable = FALSE;
  config->corner_servo_enable = TRUE;
  config->corner_cycle_count = 1;
  config->sequence_repeat_count = 1;
  config->corner_offset_h_deg = 15.0;
  config->corner_offset_v_deg = 10.0;
  config->corner_dwell_ms = 1000;
  config->corner_servo_speed = 30;
  config->corner_zoom_in_focal = NAN;
  config->corner_zoom_out_focal = NAN;
  config->zoom_in_duration_ms = 1000;
  config->zoom_out_duration_ms = 1000;
  config->corner_home_loc_h_deg = NAN;
  config->corner_home_loc_v_deg = NAN;
  config->startup_pt_focal_min_enable = FALSE;
  config->startup_pt_focal_min_hold_ms = 3000;
  config->corner_home_pt_focus = G_MAXUINT;

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
    } else if (paramKey == "debug") {
      config->debug = itr->second.as<gboolean>();
    } else if (paramKey == "print-upstream-state") {
      config->print_upstream_state = itr->second.as<gboolean>();
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
    } else if (paramKey == "auto-track-enable") {
      config->auto_track_enable = itr->second.as<gboolean>();
    } else if (paramKey == "control-source-id") {
      config->control_source_id = itr->second.as<guint>();
    } else if (paramKey == "control-period-ms") {
      config->control_period_ms = itr->second.as<guint>();
    } else if (paramKey == "target-lost-hold-ms") {
      config->target_lost_hold_ms = itr->second.as<guint>();
    } else if (paramKey == "tracking-history-size") {
      config->tracking_history_size = itr->second.as<guint>();
    } else if (paramKey == "center-deadband-x") {
      config->center_deadband_x = itr->second.as<gdouble>();
    } else if (paramKey == "center-deadband-y") {
      config->center_deadband_y = itr->second.as<gdouble>();
    } else if (paramKey == "servo-kp-x") {
      config->servo_kp_x = itr->second.as<gdouble>();
    } else if (paramKey == "servo-kp-y") {
      config->servo_kp_y = itr->second.as<gdouble>();
    } else if (paramKey == "servo-kv-x") {
      config->servo_kv_x = itr->second.as<gdouble>();
    } else if (paramKey == "servo-kv-y") {
      config->servo_kv_y = itr->second.as<gdouble>();
    } else if (paramKey == "servo-dir-x") {
      config->servo_dir_x = itr->second.as<gint>();
    } else if (paramKey == "servo-dir-y") {
      config->servo_dir_y = itr->second.as<gint>();
    } else if (paramKey == "servo-max-step-h") {
      config->servo_max_step_h = itr->second.as<gdouble>();
    } else if (paramKey == "servo-max-step-v") {
      config->servo_max_step_v = itr->second.as<gdouble>();
    } else if (paramKey == "servo-focal-adaptive-enable") {
      config->servo_focal_adaptive_enable = itr->second.as<gboolean>();
    } else if (paramKey == "servo-focal-max-step-scale-min") {
      config->servo_focal_max_step_scale_min = itr->second.as<gdouble>();
    } else if (paramKey == "servo-focal-speed-scale-min") {
      config->servo_focal_speed_scale_min = itr->second.as<gdouble>();
    } else if (paramKey == "servo-min-speed") {
      config->servo_min_speed = itr->second.as<guint>();
    } else if (paramKey == "servo-max-speed") {
      config->servo_max_speed = itr->second.as<guint>();
    } else if (paramKey == "zoom-target-ratio-min") {
      config->zoom_target_ratio_min = itr->second.as<gdouble>();
    } else if (paramKey == "zoom-target-ratio-max") {
      config->zoom_target_ratio_max = itr->second.as<gdouble>();
    } else if (paramKey == "zoom-deadband") {
      config->zoom_deadband = itr->second.as<gdouble>();
    } else if (paramKey == "visible-focal-hold-ms") {
      config->visible_focal_hold_ms = itr->second.as<guint>();
    } else if (paramKey == "visible-light-control-enable") {
      config->visible_light_control_enable = itr->second.as<gboolean>();
    } else if (paramKey == "servo-dev-id") {
      config->servo_dev_id = itr->second.as<guint>();
    } else if (paramKey == "pt-focal-min") {
      config->pt_focal_min = itr->second.as<gdouble>();
    } else if (paramKey == "pt-focal-max") {
      config->pt_focal_max = itr->second.as<gdouble>();
    } else if (paramKey == "servo-effect-threshold-h") {
      config->servo_effect_threshold_h = itr->second.as<gdouble>();
    } else if (paramKey == "servo-effect-threshold-v") {
      config->servo_effect_threshold_v = itr->second.as<gdouble>();
    } else if (paramKey == "state-stale-timeout-ms") {
      config->state_stale_timeout_ms = itr->second.as<guint>();
    } else if (paramKey == "corner-zoom-cycle-enable") {
      config->corner_zoom_cycle_enable = itr->second.as<gboolean>();
    } else if (paramKey == "corner-servo-enable") {
      config->corner_servo_enable = itr->second.as<gboolean>();
    } else if (paramKey == "corner-cycle-count") {
      config->corner_cycle_count = itr->second.as<guint>();
    } else if (paramKey == "sequence-repeat-count") {
      config->sequence_repeat_count = itr->second.as<guint>();
    } else if (paramKey == "corner-offset-h-deg") {
      config->corner_offset_h_deg = itr->second.as<gdouble>();
    } else if (paramKey == "corner-offset-v-deg") {
      config->corner_offset_v_deg = itr->second.as<gdouble>();
    } else if (paramKey == "corner-dwell-ms") {
      config->corner_dwell_ms = itr->second.as<guint>();
    } else if (paramKey == "corner-servo-speed") {
      config->corner_servo_speed = itr->second.as<guint>();
    } else if (paramKey == "corner-zoom-in-focal") {
      config->corner_zoom_in_focal = itr->second.as<gdouble>();
    } else if (paramKey == "corner-zoom-out-focal") {
      config->corner_zoom_out_focal = itr->second.as<gdouble>();
    } else if (paramKey == "zoom-in-duration-ms") {
      config->zoom_in_duration_ms = itr->second.as<guint>();
    } else if (paramKey == "zoom-out-duration-ms") {
      config->zoom_out_duration_ms = itr->second.as<guint>();
    } else if (paramKey == "corner-home-loc-h-deg" ||
               paramKey == "corner-return-loc-h-deg") {
      config->corner_home_loc_h_deg = itr->second.as<gdouble>();
    } else if (paramKey == "corner-home-loc-v-deg" ||
               paramKey == "corner-return-loc-v-deg") {
      config->corner_home_loc_v_deg = itr->second.as<gdouble>();
    } else if (paramKey == "startup-pt-focal-min-enable") {
      config->startup_pt_focal_min_enable = itr->second.as<gboolean>();
    } else if (paramKey == "startup-pt-focal-min-hold-ms") {
      config->startup_pt_focal_min_hold_ms = itr->second.as<guint>();
    } else if (paramKey == "corner-home-pt-focus") {
      config->corner_home_pt_focus = itr->second.as<guint>();
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
