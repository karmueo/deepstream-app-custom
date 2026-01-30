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

#ifndef __DEEPSTREAM_APP_PROBES_H__
#define __DEEPSTREAM_APP_PROBES_H__

#include <gst/gst.h>

#include "deepstream_app.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief 主推理完成后的探针回调，用于处理主推理输出与元数据。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param u_data 用户数据，通常为 AppCtx 指针。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn gie_primary_processing_done_buf_prob(GstPad *pad,
                                                       GstPadProbeInfo *info,
                                                       gpointer u_data);

/**
 * @brief 全部推理（主+次）完成后的探针回调，进入 OSD 或 sink 前执行。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param u_data 用户数据，通常为 NvDsInstanceBin 指针。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn gie_processing_done_buf_prob(GstPad *pad,
                                               GstPadProbeInfo *info,
                                               gpointer u_data);

/**
 * @brief 跟踪与分析完成后的探针回调，用于后处理与输出。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param u_data 用户数据，通常为 NvDsInstanceBin 指针。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn analytics_done_buf_prob(GstPad *pad,
                                          GstPadProbeInfo *info,
                                          gpointer u_data);

/**
 * @brief 统计端到端延迟的探针回调（用于主 pipeline）。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param u_data 用户数据，通常为 AppCtx 指针。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn latency_measurement_buf_prob(GstPad *pad,
                                               GstPadProbeInfo *info,
                                               gpointer u_data);

/**
 * @brief 统计 demux 分支延迟的探针回调。
 *
 * @param pad Pad 指针。
 * @param info Probe 信息。
 * @param u_data 用户数据，通常为 AppCtx 指针。
 * @return GstPadProbeReturn Probe 处理结果。
 */
GstPadProbeReturn demux_latency_measurement_buf_prob(GstPad *pad,
                                                     GstPadProbeInfo *info,
                                                     gpointer u_data);

#ifdef __cplusplus
}
#endif

#endif  // __DEEPSTREAM_APP_PROBES_H__
