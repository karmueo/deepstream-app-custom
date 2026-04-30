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
#include <string>
#include <cstring>
#include <iostream>

using std::cout;
using std::endl;

gboolean
parse_sink_yaml(NvDsSinkSubBinConfig *config, std::string group_str, gchar *cfg_file_path)
{
    gboolean ret = FALSE;
    YAML::Node configyml = YAML::LoadFile(cfg_file_path);
    YAML::Node sink_node = configyml[group_str];

    config->encoder_config.rtsp_port = 8554;
    config->encoder_config.udp_port = 5000;
    config->encoder_config.codec = NV_DS_ENCODER_H264;
    config->encoder_config.container = NV_DS_CONTAINER_MP4;
    config->encoder_config.compute_hw = 0;
    config->render_config.qos = FALSE;
    config->link_to_demux = FALSE;
    config->msg_conv_broker_config.new_api = FALSE;
    config->msg_conv_broker_config.conv_msg2p_new_api = FALSE;
    config->msg_conv_broker_config.conv_frame_interval = 30;
    config->cuav_control_config.enable = FALSE;
    config->cuav_control_config.multicast_ip = NULL;
    config->cuav_control_config.port = 18003;
    config->cuav_control_config.iface = NULL;
    config->cuav_control_config.ttl = 1;
    config->cuav_control_config.debug = FALSE;
    config->cuav_control_config.print_upstream_state = FALSE;
    config->cuav_control_config.tx_sys_id = 999;
    config->cuav_control_config.tx_dev_type = 1;
    config->cuav_control_config.tx_dev_id = 999;
    config->cuav_control_config.tx_subdev_id = 999;
    config->cuav_control_config.rx_sys_id = 999;
    config->cuav_control_config.rx_dev_type = 1;
    config->cuav_control_config.rx_dev_id = 999;
    config->cuav_control_config.rx_subdev_id = 999;
    config->cuav_control_config.send_test_on_startup = FALSE;
    config->cuav_control_config.auto_track_enable = FALSE;
    config->cuav_control_config.control_source_id = 0;
    config->cuav_control_config.control_period_ms = 100;
    config->cuav_control_config.target_lost_hold_ms = 1000;
    config->cuav_control_config.tracking_history_size = 5;
    config->cuav_control_config.center_deadband_x = 0.03;
    config->cuav_control_config.center_deadband_y = 0.03;
    config->cuav_control_config.servo_kp_x = 1.5;
    config->cuav_control_config.servo_kp_y = 1.5;
    config->cuav_control_config.servo_kv_x = 0.35;
    config->cuav_control_config.servo_kv_y = 0.35;
    config->cuav_control_config.servo_dir_x = 1;
    config->cuav_control_config.servo_dir_y = -1;
    config->cuav_control_config.servo_max_step_h = 1.5;
    config->cuav_control_config.servo_max_step_v = 1.0;
    config->cuav_control_config.servo_focal_adaptive_enable = TRUE;
    config->cuav_control_config.servo_focal_max_step_scale_min = 0.25;
    config->cuav_control_config.servo_focal_speed_scale_min = 0.50;
    config->cuav_control_config.servo_min_speed = 10;
    config->cuav_control_config.servo_max_speed = 60;
    config->cuav_control_config.zoom_target_ratio_min = 0.20;
    config->cuav_control_config.zoom_target_ratio_max = 0.35;
    config->cuav_control_config.zoom_deadband = 0.02;
    config->cuav_control_config.visible_focal_hold_ms = 300;
    config->cuav_control_config.visible_light_control_enable = TRUE;
    config->cuav_control_config.servo_dev_id = 2;
    config->cuav_control_config.pt_focal_min = 134.0;
    config->cuav_control_config.pt_focal_max = 16298.0;
    config->cuav_control_config.servo_effect_threshold_h = 0.5;
    config->cuav_control_config.servo_effect_threshold_v = 0.3;
    config->cuav_control_config.state_stale_timeout_ms = 2000;
    config->cuav_control_config.corner_zoom_cycle_enable = FALSE;
    config->cuav_control_config.corner_cycle_count = 1;
    config->cuav_control_config.sequence_repeat_count = 1;
    config->cuav_control_config.corner_offset_h_deg = 15.0;
    config->cuav_control_config.corner_offset_v_deg = 10.0;
    config->cuav_control_config.corner_dwell_ms = 1000;
    config->cuav_control_config.corner_servo_speed = 30;
    config->cuav_control_config.corner_home_loc_h_deg = NAN;
    config->cuav_control_config.corner_home_loc_v_deg = NAN;
    config->cuav_control_config.startup_pt_focal_min_enable = FALSE;
    config->cuav_control_config.startup_pt_focal = 20.0;
    config->cuav_control_config.startup_pt_focus = 100;
    config->cuav_control_config.lost_target_focal_min_hold_ms = 3000;
    config->cuav_control_config.corner_home_pt_focus = G_MAXUINT;

    if (sink_node["enable"])
    {
        gboolean val = sink_node["enable"].as<gboolean>();
        if (val == FALSE)
            return TRUE;
    }

    if (sink_node["cuav-control-config-file"])
    {
        std::string temp = sink_node["cuav-control-config-file"].as<std::string>();
        char *str = (char *)malloc(sizeof(char) * 1024);
        char *abs_cfg_path = (char *)malloc(sizeof(char) * 1024);
        std::strncpy(str, temp.c_str(), 1023);
        str[1023] = '\0';
        if (!get_absolute_file_path_yaml(cfg_file_path, str, abs_cfg_path))
        {
            g_printerr("Error: Could not resolve cuav-control-config-file in sink.\n");
            g_free(str);
            g_free(abs_cfg_path);
            goto done;
        }
        g_free(str);

        if (!parse_cuav_control_yaml(&config->cuav_control_config, abs_cfg_path))
        {
            g_printerr("Error: Could not parse cuav-control-config-file '%s'.\n",
                       abs_cfg_path);
            g_free(abs_cfg_path);
            goto done;
        }
        g_free(abs_cfg_path);
    }

    for (YAML::const_iterator itr = sink_node.begin();
         itr != sink_node.end(); ++itr)
    {
        std::string paramKey = itr->first.as<std::string>();

        if (paramKey == "enable")
        {
            config->enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "type")
        {
            config->type =
                (NvDsSinkType)itr->second.as<int>();
        }
        else if (paramKey == "link-to-demux")
        {
            config->link_to_demux = itr->second.as<gboolean>();
        }
        else if (paramKey == "width")
        {
            config->render_config.width = itr->second.as<gint>();
        }
        else if (paramKey == "height")
        {
            config->render_config.height = itr->second.as<gint>();
        }
        else if (paramKey == "qos")
        {
            config->render_config.qos = itr->second.as<gboolean>();
            config->render_config.qos_value_specified = TRUE;
        }
        else if (paramKey == "sync")
        {
            config->sync = itr->second.as<gint>();
        }
        else if (paramKey == "nvbuf-memory-type")
        {
            config->render_config.nvbuf_memory_type =
                itr->second.as<guint>();
        }
        else if (paramKey == "container")
        {
            config->encoder_config.container =
                (NvDsContainerType)itr->second.as<int>();
        }
        else if (paramKey == "codec")
        {
            config->encoder_config.codec =
                (NvDsEncoderType)itr->second.as<int>();
        }
        else if (paramKey == "compute-hw")
        {
            config->encoder_config.compute_hw =
                itr->second.as<int>();
        }
        else if (paramKey == "enc-type")
        {
            config->encoder_config.enc_type =
                (NvDsEncHwSwType)itr->second.as<int>();
        }
        else if (paramKey == "bitrate")
        {
            config->encoder_config.bitrate =
                itr->second.as<gint>();
        }
        else if (paramKey == "profile")
        {
            config->encoder_config.profile =
                itr->second.as<guint>();
        }
        else if (paramKey == "iframeinterval")
        {
            config->encoder_config.iframeinterval =
                itr->second.as<guint>();
        }
        else if (paramKey == "output-file")
        {
            std::string temp = itr->second.as<std::string>();
            config->encoder_config.output_file_path = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->encoder_config.output_file_path, temp.c_str(), 1023);
        }
        else if (paramKey == "source-id")
        {
            config->source_id =
                itr->second.as<guint>();
        }
        else if (paramKey == "rtsp-port")
        {
            config->encoder_config.rtsp_port =
                itr->second.as<guint>();
        }
        else if (paramKey == "udp-port")
        {
            config->encoder_config.udp_port =
                itr->second.as<guint>();
        }
        else if (paramKey == "udp-buffer-size")
        {
            config->encoder_config.udp_buffer_size =
                itr->second.as<guint64>();
        }
        else if (paramKey == "color-range")
        {
            config->render_config.color_range =
                itr->second.as<guint>();
        }
        else if (paramKey == "conn-id")
        {
            config->render_config.conn_id =
                itr->second.as<guint>();
        }
        else if (paramKey == "plane-id")
        {
            config->render_config.plane_id =
                itr->second.as<guint>();
        }
        else if (paramKey == "set-mode")
        {
            config->render_config.set_mode =
                itr->second.as<gboolean>();
        }
        else if (paramKey == "gpu-id")
        {
            config->encoder_config.gpu_id = config->render_config.gpu_id =
                itr->second.as<guint>();
        }
        else if (paramKey == "msg-conv-config" ||
                 paramKey == "msg-conv-payload-type" ||
                 paramKey == "msg-conv-msg2p-lib" ||
                 paramKey == "msg-conv-comp-id" ||
                 paramKey == "debug-payload-dir" ||
                 paramKey == "multiple-payloads" ||
                 paramKey == "msg-conv-msg2p-new-api" ||
                 paramKey == "msg-conv-frame-interval" ||
                 paramKey == "msg-conv-dummy-payload")
        {
            ret = parse_msgconv_yaml(&config->msg_conv_broker_config, group_str, cfg_file_path);
            if (!ret)
                goto done;
        }
        else if (paramKey == "msg-broker-proto-lib")
        {
            std::string temp = itr->second.as<std::string>();
            config->msg_conv_broker_config.proto_lib = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->msg_conv_broker_config.proto_lib, temp.c_str(), 1023);
        }
        else if (paramKey == "msg-broker-conn-str")
        {
            std::string temp = itr->second.as<std::string>();
            config->msg_conv_broker_config.conn_str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->msg_conv_broker_config.conn_str, temp.c_str(), 1023);
        }
        else if (paramKey == "topic")
        {
            std::string temp = itr->second.as<std::string>();
            config->msg_conv_broker_config.topic = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->msg_conv_broker_config.topic, temp.c_str(), 1023);
        }
        else if (paramKey == "msg-broker-config")
        {
            std::string temp = itr->second.as<std::string>();
            char *str = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(str, temp.c_str(), 1023);
            config->msg_conv_broker_config.broker_config_file_path = (char *)malloc(sizeof(char) * 1024);
            if (!get_absolute_file_path_yaml(cfg_file_path, str,
                                             config->msg_conv_broker_config.broker_config_file_path))
            {
                g_printerr("Error: Could not parse msg-broker-config in sink.\n");
                g_free(str);
                goto done;
            }
            g_free(str);
        }
        else if (paramKey == "msg-broker-comp-id")
        {
            config->msg_conv_broker_config.broker_comp_id =
                itr->second.as<guint>();
        }
        else if (paramKey == "disable-msgconv")
        {
            config->msg_conv_broker_config.disable_msgconv =
                itr->second.as<gboolean>();
        }
        else if (paramKey == "new-api")
        {
            config->msg_conv_broker_config.new_api =
                itr->second.as<gboolean>();
        }
        else if (paramKey == "cuav-control-config-file")
        {
            continue;
        }
        else if (paramKey == "ip")
        {
            // 设置报文发送组播地址
            std::string temp = itr->second.as<std::string>();
            config->mynetwork_config.ip = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->mynetwork_config.ip, temp.c_str(), 1023);
        }
        else if (paramKey == "multicast-port")
        {
            // 设置报文发送组播端口
            config->mynetwork_config.multicast_port =
                itr->second.as<gint>();
        }
        else if (paramKey == "multicast-iface")
        {
            // 设置报文发送组播网卡
            std::string temp = itr->second.as<std::string>();
            config->mynetwork_config.iface = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->mynetwork_config.iface, temp.c_str(), 1023);
        }
        else if (paramKey == "fps")
        {
            // 设置帧率
            config->mynetwork_config.fps =
                itr->second.as<gint>();
        }
        else if (paramKey == "multicast-ip")
        {
            std::string temp = itr->second.as<std::string>();
            config->cuav_control_config.multicast_ip = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->cuav_control_config.multicast_ip, temp.c_str(), 1023);
            config->cuav_control_config.multicast_ip[1023] = '\0';
        }
        else if (paramKey == "port")
        {
            config->cuav_control_config.port = itr->second.as<gint>();
        }
        else if (paramKey == "interface")
        {
            std::string temp = itr->second.as<std::string>();
            config->cuav_control_config.iface = (char *)malloc(sizeof(char) * 1024);
            std::strncpy(config->cuav_control_config.iface, temp.c_str(), 1023);
            config->cuav_control_config.iface[1023] = '\0';
        }
        else if (paramKey == "ttl")
        {
            config->cuav_control_config.ttl = itr->second.as<guint>();
        }
        else if (paramKey == "debug")
        {
            config->cuav_control_config.debug = itr->second.as<gboolean>();
        }
        else if (paramKey == "print-upstream-state")
        {
            config->cuav_control_config.print_upstream_state = itr->second.as<gboolean>();
        }
        else if (paramKey == "tx-sys-id")
        {
            config->cuav_control_config.tx_sys_id = itr->second.as<guint>();
        }
        else if (paramKey == "tx-dev-type")
        {
            config->cuav_control_config.tx_dev_type = itr->second.as<guint>();
        }
        else if (paramKey == "tx-dev-id")
        {
            config->cuav_control_config.tx_dev_id = itr->second.as<guint>();
        }
        else if (paramKey == "tx-subdev-id")
        {
            config->cuav_control_config.tx_subdev_id = itr->second.as<guint>();
        }
        else if (paramKey == "rx-sys-id")
        {
            config->cuav_control_config.rx_sys_id = itr->second.as<guint>();
        }
        else if (paramKey == "rx-dev-type")
        {
            config->cuav_control_config.rx_dev_type = itr->second.as<guint>();
        }
        else if (paramKey == "rx-dev-id")
        {
            config->cuav_control_config.rx_dev_id = itr->second.as<guint>();
        }
        else if (paramKey == "rx-subdev-id")
        {
            config->cuav_control_config.rx_subdev_id = itr->second.as<guint>();
        }
        else if (paramKey == "send-test-on-startup")
        {
            config->cuav_control_config.send_test_on_startup = itr->second.as<gboolean>();
        }
        else if (paramKey == "auto-track-enable")
        {
            config->cuav_control_config.auto_track_enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "control-source-id")
        {
            config->cuav_control_config.control_source_id = itr->second.as<guint>();
        }
        else if (paramKey == "control-period-ms")
        {
            config->cuav_control_config.control_period_ms = itr->second.as<guint>();
        }
        else if (paramKey == "target-lost-hold-ms")
        {
            config->cuav_control_config.target_lost_hold_ms = itr->second.as<guint>();
        }
        else if (paramKey == "tracking-history-size")
        {
            config->cuav_control_config.tracking_history_size = itr->second.as<guint>();
        }
        else if (paramKey == "center-deadband-x")
        {
            config->cuav_control_config.center_deadband_x = itr->second.as<gdouble>();
        }
        else if (paramKey == "center-deadband-y")
        {
            config->cuav_control_config.center_deadband_y = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-kp-x")
        {
            config->cuav_control_config.servo_kp_x = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-kp-y")
        {
            config->cuav_control_config.servo_kp_y = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-kv-x")
        {
            config->cuav_control_config.servo_kv_x = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-kv-y")
        {
            config->cuav_control_config.servo_kv_y = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-dir-x")
        {
            config->cuav_control_config.servo_dir_x = itr->second.as<gint>();
        }
        else if (paramKey == "servo-dir-y")
        {
            config->cuav_control_config.servo_dir_y = itr->second.as<gint>();
        }
        else if (paramKey == "servo-max-step-h")
        {
            config->cuav_control_config.servo_max_step_h = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-max-step-v")
        {
            config->cuav_control_config.servo_max_step_v = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-focal-adaptive-enable")
        {
            config->cuav_control_config.servo_focal_adaptive_enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "servo-focal-max-step-scale-min")
        {
            config->cuav_control_config.servo_focal_max_step_scale_min = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-focal-speed-scale-min")
        {
            config->cuav_control_config.servo_focal_speed_scale_min = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-min-speed")
        {
            config->cuav_control_config.servo_min_speed = itr->second.as<guint>();
        }
        else if (paramKey == "servo-max-speed")
        {
            config->cuav_control_config.servo_max_speed = itr->second.as<guint>();
        }
        else if (paramKey == "zoom-target-ratio-min")
        {
            config->cuav_control_config.zoom_target_ratio_min = itr->second.as<gdouble>();
        }
        else if (paramKey == "zoom-target-ratio-max")
        {
            config->cuav_control_config.zoom_target_ratio_max = itr->second.as<gdouble>();
        }
        else if (paramKey == "zoom-deadband")
        {
            config->cuav_control_config.zoom_deadband = itr->second.as<gdouble>();
        }
        else if (paramKey == "visible-focal-hold-ms")
        {
            config->cuav_control_config.visible_focal_hold_ms = itr->second.as<guint>();
        }
        else if (paramKey == "visible-light-control-enable")
        {
            config->cuav_control_config.visible_light_control_enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "servo-dev-id")
        {
            config->cuav_control_config.servo_dev_id = itr->second.as<guint>();
        }
        else if (paramKey == "pt-focal-min")
        {
            config->cuav_control_config.pt_focal_min = itr->second.as<gdouble>();
        }
        else if (paramKey == "pt-focal-max")
        {
            config->cuav_control_config.pt_focal_max = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-effect-threshold-h")
        {
            config->cuav_control_config.servo_effect_threshold_h = itr->second.as<gdouble>();
        }
        else if (paramKey == "servo-effect-threshold-v")
        {
            config->cuav_control_config.servo_effect_threshold_v = itr->second.as<gdouble>();
        }
        else if (paramKey == "state-stale-timeout-ms")
        {
            config->cuav_control_config.state_stale_timeout_ms = itr->second.as<guint>();
        }
        else if (paramKey == "corner-zoom-cycle-enable")
        {
            config->cuav_control_config.corner_zoom_cycle_enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "corner-cycle-count")
        {
            config->cuav_control_config.corner_cycle_count = itr->second.as<guint>();
        }
        else if (paramKey == "sequence-repeat-count")
        {
            config->cuav_control_config.sequence_repeat_count = itr->second.as<guint>();
        }
        else if (paramKey == "corner-offset-h-deg")
        {
            config->cuav_control_config.corner_offset_h_deg = itr->second.as<gdouble>();
        }
        else if (paramKey == "corner-offset-v-deg")
        {
            config->cuav_control_config.corner_offset_v_deg = itr->second.as<gdouble>();
        }
        else if (paramKey == "corner-dwell-ms")
        {
            config->cuav_control_config.corner_dwell_ms = itr->second.as<guint>();
        }
        else if (paramKey == "corner-servo-speed")
        {
            config->cuav_control_config.corner_servo_speed = itr->second.as<guint>();
        }
        else if (paramKey == "corner-home-loc-h-deg" ||
                 paramKey == "corner-return-loc-h-deg")
        {
            config->cuav_control_config.corner_home_loc_h_deg = itr->second.as<gdouble>();
        }
        else if (paramKey == "corner-home-loc-v-deg" ||
                 paramKey == "corner-return-loc-v-deg")
        {
            config->cuav_control_config.corner_home_loc_v_deg = itr->second.as<gdouble>();
        }
        else if (paramKey == "startup-pt-focal-min-enable")
        {
            config->cuav_control_config.startup_pt_focal_min_enable = itr->second.as<gboolean>();
        }
        else if (paramKey == "startup-pt-focal")
        {
            config->cuav_control_config.startup_pt_focal = itr->second.as<gdouble>();
        }
        else if (paramKey == "startup-pt-focus")
        {
            config->cuav_control_config.startup_pt_focus = itr->second.as<guint>();
        }
        else if (paramKey == "lost-target-focal-min-hold-ms")
        {
            config->cuav_control_config.lost_target_focal_min_hold_ms = itr->second.as<guint>();
        }
        else if (paramKey == "corner-home-pt-focus")
        {
            config->cuav_control_config.corner_home_pt_focus = itr->second.as<guint>();
        }
        else
        {
            cout << "[WARNING] Unknown param found in sink: " << paramKey << endl;
        }
    }

    ret = TRUE;
done:
    if (!ret)
    {
        cout << __func__ << " failed" << endl;
    }
    return ret;
}
