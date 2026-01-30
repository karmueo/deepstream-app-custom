/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVGSTDS_APP_H__
#define __NVGSTDS_APP_H__

#include <gst/gst.h>
#include <stdio.h>

#include "deepstream_app_version.h"
#include "deepstream_c2d_msg.h"
#include "deepstream_common.h"
#include "deepstream_config.h"
#include "deepstream_dsanalytics.h"
#include "deepstream_dsexample.h"
#include "deepstream_image_save.h"
#include "deepstream_osd.h"
#include "deepstream_perf.h"
#include "deepstream_preprocess.h"
#include "deepstream_primary_gie.h"
#include "deepstream_secondary_gie.h"
#include "deepstream_secondary_preprocess.h"
#include "deepstream_segvisual.h"
#include "deepstream_sinks.h"
#include "deepstream_sources.h"
#include "deepstream_streammux.h"
#include "deepstream_tiled_display.h"
#include "deepstream_tracker.h"
#include "deepstream_udpmulticast.h"
#include "deepstream_udpjsonmeta.h"
#include "deepstream_videorecognition.h"
#include "gst-nvdscommonconfig.h"
#include "gst-nvdscustommessage.h"
#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct _AppCtx AppCtx;

/**
 * @brief 边界框生成回调函数类型定义。
 *
 * 在完成推理和分析后调用此回调，用于处理检测到的目标边界框信息。
 *
 * @param appCtx 应用程序上下文指针。
 * @param buf GStreamer 缓冲区指针。
 * @param batch_meta 批处理元数据指针，包含检测结果。
 * @param index 源索引。
 */
typedef void (*bbox_generated_callback)(AppCtx *appCtx, GstBuffer *buf,
                                            NvDsBatchMeta *batch_meta,
                                            guint          index);
/**
 * @brief 图形叠加回调函数类型定义。
 *
 * 用于在帧上叠加自定义图形（如边界框标签、标注等）。
 *
 * @param appCtx 应用程序上下文指针。
 * @param buf GStreamer 缓冲区指针。
 * @param batch_meta 批处理元数据指针。
 * @param index 源索引。
 * @return gboolean 叠加成功返回 TRUE，否则返回 FALSE。
 */
typedef gboolean (*overlay_graphics_callback)(AppCtx        *appCtx,
                                                  GstBuffer     *buf,
                                                  NvDsBatchMeta *batch_meta,
                                                  guint          index);

/**
 * @brief 单路视频流的处理单元，包含该流的完整处理组件。
 *
 * 每个源视频流对应一个 NvDsInstanceBin，包含从预处理到输出的所有处理组件。
 */
typedef struct
{
    guint                      index;                           /**< 源索引，标识该实例对应的视频流 */
    gulong                     all_bbox_buffer_probe_id;        /**< 所有边界框缓冲探针 ID */
    gulong                     primary_bbox_buffer_probe_id;    /**< 主推理边界框缓冲探针 ID */
    gulong                     fps_buffer_probe_id;             /**< FPS 缓冲探针 ID */
    GstElement                *bin;                             /**< 该实例的 GStreamer bin 元素 */
    GstElement                *tee;                             /**< TEE 元素，用于分流处理 */
    GstElement                *msg_conv;                        /**< 消息转换器元素 */
    NvDsPreProcessBin          preprocess_bin;                  /**< 预处理 bin */
    NvDsPrimaryGieBin          primary_gie_bin;                 /**< 主推理 bin (YOLO) */
    NvDsOSDBin                 osd_bin;                         /**< OSD (On-Screen Display) bin */
    NvDsSegVisualBin           segvisual_bin;                   /**< 分割可视化 bin */
    NvDsSecondaryGieBin        secondary_gie_bin;               /**< 次级推理 bin */
    NvDsSecondaryPreProcessBin secondary_preprocess_bin;        /**< 次级预处理 bin */
    NvDsTrackerBin             tracker_bin;                     /**< 跟踪器 bin (SOT) */
    NvDsSinkBin                sink_bin;                        /**< 输出 sink bin */
    NvDsSinkBin                demux_sink_bin;                  /**< 解复用输出 sink bin */
    NvDsDsAnalyticsBin         dsanalytics_bin;                 /**< DS Analytics bin */
    NvDsDsExampleBin           dsexample_bin;                   /**< Dsexample bin */
    NvDsVideoRecognitionBin    videorecognition_bin;            /**< 视频识别 bin (X3D 动作识别) */
    GstElement                *udpjsonmeta;                    /**< UDP JSON 元数据插件 */
    AppCtx                    *appCtx;                          /**< 指向应用程序上下文的指针 */
} NvDsInstanceBin;

/**
 * @brief 视频分析流水线结构体，管理整个 GStreamer 流水线的核心组件。
 *
 * 包含多路视频源、实例 bin、tiled display、demuxer 等核心组件。
 */
typedef struct
{
    gulong              primary_bbox_buffer_probe_id;          /**< 主推理边界框缓冲探针 ID */
    guint               bus_id;                                /**< 总线 ID */
    GstElement         *pipeline;                              /**< 主 GStreamer pipeline 元素 */
    NvDsSrcParentBin    multi_src_bin;                         /**< 多路视频源 bin */
    NvDsInstanceBin     instance_bins[MAX_SOURCE_BINS];        /**< 各视频流实例 bin 数组 */
    NvDsInstanceBin     demux_instance_bins[MAX_SOURCE_BINS];  /**< 解复用分支实例 bin 数组 */
    NvDsInstanceBin     common_elements;                       /**< 公共元素实例 bin */
    GstElement         *tiler_tee;                             /**< Tiled display TEE 元素 */
    NvDsTiledDisplayBin tiled_display_bin;                     /**< Tiled display bin */
    GstElement         *demuxer;                               /**< 流解复用器元素 */
    NvDsDsExampleBin    dsexample_bin;                         /**< Dsexample bin */
    AppCtx             *appCtx;                                /**< 指向应用程序上下文的指针 */
} NvDsPipeline;

/**
 * @brief 应用程序配置结构体，存储所有组件的配置参数。
 *
 * 从配置文件加载并解析所有组件的配置信息，包括视频源、推理模型、跟踪器、输出等。
 */
typedef struct
{
    /** 性能与运行控制配置 */
    gboolean enable_perf_measurement;                          /**< 是否启用性能测量 */
    gint     file_loop;                                        /**< 文件循环播放次数 (-1 为无限循环) */
    gint     pipeline_recreate_sec;                            /**< 流水线重建间隔（秒），0 为不重建 */
    gboolean source_list_enabled;                              /**< 是否启用源列表模式 */
    guint    total_num_sources;                                /**< 总源数量 */
    guint    num_source_sub_bins;                              /**< 源 sub-bin 数量 */
    guint    num_secondary_gie_sub_bins;                       /**< 次级推理 sub-bin 数量 */
    guint    num_secondary_preprocess_sub_bins;                /**< 次级预处理 sub-bin 数量 */
    guint    num_sink_sub_bins;                                /**< sink sub-bin 数量 */
    guint    num_message_consumers;                            /**< 消息消费者数量 */
    guint    perf_measurement_interval_sec;                    /**< 性能测量间隔（秒） */
    guint    sgie_batch_size;                                  /**< 次级推理批处理大小 */
    gboolean extract_sei_type5_data;                           /**< 是否提取 SEI type5 数据 */
    gchar   *sei_uuid;                                         /**< SEI UUID 字符串 */
    gboolean low_latency_mode;                                 /**< 是否启用低延迟模式 */
    gchar   *bbox_dir_path;                                    /**< 边界框输出目录路径 */
    gchar   *kitti_track_dir_path;                             /**< KITTI 跟踪输出目录路径 */
    gchar   *reid_track_dir_path;                              /**< ReID 跟踪输出目录路径 */
    gchar   *terminated_track_output_path;                     /**< 终止跟踪输出文件路径 */
    gchar   *shadow_track_output_path;                         /**< 影子跟踪输出文件路径 */
    gboolean enable_jpeg_save;                                 /**< 是否启用运行时 JPEG 保存 */
    gboolean detect_record_default_enable;                     /**< 检测触发录制的初始状态 */

    /** 源配置数组 */
    gchar              **uri_list;                             /**< 视频 URI 列表 */
    gchar              **sensor_id_list;                       /**< 传感器 ID 列表 */
    gchar              **sensor_name_list;                     /**< 传感器名称列表 */
    NvDsSourceConfig     multi_source_config[MAX_SOURCE_BINS]; /**< 多源配置数组 */

    /** 各组件配置 */
    NvDsStreammuxConfig  streammux_config;                     /**< Streammux 配置 */
    NvDsOSDConfig        osd_config;                           /**< OSD 配置 */
    NvDsSegVisualConfig  segvisual_config;                     /**< 分割可视化配置 */
    NvDsPreProcessConfig preprocess_config;                    /**< 预处理配置 */
    NvDsPreProcessConfig
        secondary_preprocess_sub_bin_config[MAX_SECONDARY_PREPROCESS_BINS]; /**< 次级预处理配置数组 */
    NvDsGieConfig     primary_gie_config;                      /**< 主推理配置 */
    NvDsTrackerConfig tracker_config;                          /**< 跟踪器配置 */
    NvDsGieConfig     secondary_gie_sub_bin_config[MAX_SECONDARY_GIE_BINS]; /**< 次级推理配置数组 */
    NvDsSinkSubBinConfig   sink_bin_sub_bin_config[MAX_SINK_BINS]; /**< Sink 配置数组 */
    NvDsMsgConsumerConfig  message_consumer_config[MAX_MESSAGE_CONSUMERS]; /**< 消息消费者配置数组 */
    NvDsTiledDisplayConfig tiled_display_config;               /**< Tiled display 配置 */
    NvDsDsAnalyticsConfig  dsanalytics_config;                 /**< DS Analytics 配置 */
    NvDsDsExampleConfig    dsexample_config;                   /**< Dsexample 配置 */
    NvDsVideoRecognitionConfig videorecognition_config;        /**< 视频识别配置 (X3D) */
    NvDsUdpMulticastConfig udpmulticast_config;               /**< UDP 多播配置 */
    NvDsUdpJsonMetaConfig  udpjsonmeta_config;                /**< UDP JSON 元数据配置 */
    NvDsSinkMsgConvBrokerConfig msg_conv_config;              /**< 消息转换 broker 配置 */
    NvDsMyNetworkConfig         mynetwork_config;             /**< 自定义网络配置 */
    NvDsImageSave               image_save_config;            /**< 图像保存配置 */

    /** 多 URI 源支持配置 */
    gboolean         use_nvmultiurisrcbin;                     /**< 是否使用 nvmultiurisrcbin */
    gboolean         stream_name_display;                      /**< 是否显示流名称 */
    guint            max_batch_size;                           /**< 最大批处理大小 */
    gchar           *http_ip;                                  /**< HTTP 服务器 IP */
    gchar           *http_port;                                /**< HTTP 服务器端口 */
    gboolean         source_attr_all_parsed;                   /**< 是否已解析所有源属性 */
    NvDsSourceConfig source_attr_all_config;                   /**< 所有源属性配置 */

    /** 全局 GPU 配置 */
    gint global_gpu_id;                                        /**< 全局 GPU ID，未指定组件时使用 */
} NvDsConfig;

/**
 * @brief 单路视频流实例运行时数据结构体。
 *
 * 存储各视频流实例的运行时状态信息。
 */
typedef struct
{
    gulong frame_num;                                          /**< 当前帧号计数器 */
} NvDsInstanceData;

/**
 * @brief 自定义消息数据结构体。
 *
 * 用于存储通过消息队列传递的异步控制消息数据。
 */
typedef struct
{
    guint64 timestamp;                                         /**< 消息时间戳 */
} CustomMessageData;

/**
 * @brief 静止目标过滤状态结构体。
 *
 * 用于检测和过滤视频中的静止目标（避免误检），跟踪目标的位置变化。
 */
typedef struct
{
    gboolean active;               /**< 过滤功能是否激活 */
    gboolean has_last;             /**< 是否有上一帧目标数据 */
    guint64  last_object_id;       /**< 上一帧目标 ID */
    guint64  static_object_id;     /**< 静止目标 ID */
    guint    consecutive_count;    /**< 连续静止帧计数 */
    gfloat   last_cx;              /**< 上一帧目标中心 X 坐标 */
    gfloat   last_cy;              /**< 上一帧目标中心 Y 坐标 */
    gfloat   last_w;               /**< 上一帧目标宽度 */
    gfloat   last_h;               /**< 上一帧目标高度 */
    gfloat   static_cx;            /**< 静止目标中心 X 坐标 */
    gfloat   static_cy;            /**< 静止目标中心 Y 坐标 */
    gfloat   static_w;             /**< 静止目标宽度 */
    gfloat   static_h;             /**< 静止目标高度 */
} StaticTargetFilterState;

/**
 * @brief 应用程序上下文结构体，存储整个应用的核心状态与资源。
 *
 * 应用程序的核心数据结构，包含流水线、配置、回调函数、运行时状态等。
 */
struct _AppCtx
{
    /** 基本状态标志 */
    gboolean version;                /**< 版本信息标志 */
    gboolean cintr;                  /**< 中断信号标志 */
    gboolean show_bbox_text;         /**< 是否显示边界框文本 */
    gboolean seeking;                /**< 正在定位/跳转标志 */
    gboolean quit;                   /**< 退出标志 */
    gint     person_class_id;        /**< 人员类别 ID */
    gint     car_class_id;           /**< 车辆类别 ID */
    gint     return_value;           /**< 进程返回值 */
    guint    index;                  /**< 应用索引 */
    gint     active_source_index;    /**< 当前激活的源索引 */

    /** 同步原语 */
    GMutex app_lock;                 /**< 应用状态锁 */
    GCond  app_cond;                 /**< 应用条件变量 */

    /** 流水线与配置 */
    NvDsPipeline              pipeline;           /**< 视频分析流水线 */
    NvDsConfig                config;             /**< 当前配置 */
    NvDsConfig                override_config;    /**< 覆盖配置 */
    NvDsInstanceData          instance_data[MAX_SOURCE_BINS]; /**< 各源实例数据 */

    /** 消息与性能 */
    NvDsC2DContext           *c2d_ctx[MAX_MESSAGE_CONSUMERS]; /**< C2D 消息上下文 */
    NvDsAppPerfStructInt      perf_struct;        /**< 性能数据结构 */
    bbox_generated_callback   bbox_generated_post_analytics_cb; /**< 分析后边界框回调 */
    bbox_generated_callback   all_bbox_generated_cb; /**< 所有边界框生成回调 */
    overlay_graphics_callback overlay_graphics_cb; /**< 图形叠加回调 */
    NvDsFrameLatencyInfo     *latency_info;      /**< 延迟信息数组 */
    GMutex                    latency_lock;      /**< 延迟信息锁 */

    /** OTA 更新相关 */
    GThread                  *ota_handler_thread; /**< OTA 处理线程 */
    guint                     ota_inotify_fd;    /**< inotify 文件描述符 */
    guint                     ota_watch_desc;    /**< inotify 监视描述符 */

    /** 传感器信息缓存 (通过 REST API 添加/移除流时获取) */
    GHashTable *sensorInfoHash;  /**< 传感器信息哈希表，key 为 source_id */
    gboolean    eos_received;     /**< 是否收到 EOS 信号 */

    /** 目标编码上下文 */
    NvDsObjEncCtxHandle obj_ctx_handle; /**< 目标编码上下文句柄 */

    /** 自定义消息数据 */
    CustomMessageData *custom_msg_data; /**< 自定义消息数据指针 */

    /** 分类结果聚合 (历史平滑) */
    GHashTable *cls_agg_map; /**< 目标分类聚合哈希表，key: object_id, value: ObjClsAgg* */

    /** 标签锚点平滑 */
    GHashTable *label_anchor_map; /**< 标签锚点平滑哈希表 */

    /** 异步控制消息解析线程资源 */
    GAsyncQueue *control_msg_queue;       /**< 控制消息队列，元素为 ControlMsgItem* */
    GThread     *control_msg_thread;      /**< 后台消息解析线程 */
    gboolean     control_msg_thread_running; /**< 线程运行标志 */

    /** 单目标跟踪统计 */
    gboolean    tracker_stats_valid;      /**< 跟踪统计是否有效 */
    guint64     tracker_stats_current_id; /**< 当前跟踪目标 ID */
    GHashTable *tracker_stats_counts;     /**< 跟踪统计计数表，key: label, value: count */
    GQueue     *tracker_label_history;    /**< 最近 100 次识别标签的滑动窗口 */

    /** 检测触发录制状态 */
    gboolean    detect_record_enabled;    /**< 检测触发录制是否启用 */

    /** ROI-based NMS 配置缓存 */
    gboolean    roi_nms_enabled;          /**< ROI NMS 是否启用 */
    GArray     *roi_centers;              /**< ROI 中心点数组，扁平格式 [cx0,cy0,cx1,cy1,...] */

    /** 单目标跟踪连续性检测 */
    GstClockTime tracking_start_time;     /**< 当前目标开始连续跟踪的时间戳 (纳秒) */
    guint64      last_tracked_object_id;  /**< 上一次跟踪的目标 ID */
    gboolean     is_tracking_continuous;  /**< 是否正在连续跟踪同一目标 */

    /** 静止目标误检过滤状态 */
    StaticTargetFilterState static_target_filter_states[MAX_SOURCE_BINS]; /**< 各源静止目标过滤状态 */
};

/**
 * @brief 根据应用上下文配置创建 DeepStream 分析流水线。
 *
 * @param appCtx [IN/OUT] 应用程序上下文，提供配置信息并维护流水线资源。
 * @param bbox_generated_post_analytics_cb [IN] 分析后边界框生成回调
 *         (在 PGIE、Tracker 或最后一个 SGIE 之后触发)。
 * @param all_bbox_generated_cb [IN] 所有边界框生成回调。
 * @param perf_cb [IN] 性能回调函数。
 * @param overlay_graphics_cb [IN] 图形叠加回调函数。
 * @param msg_broker_subscribe_cb [IN] 消息 broker 订阅回调。
 * @return gboolean 创建成功返回 TRUE，否则返回 FALSE。
 */
gboolean
create_pipeline(AppCtx                   *appCtx,
                bbox_generated_callback   bbox_generated_post_analytics_cb,
                bbox_generated_callback   all_bbox_generated_cb,
                perf_callback             perf_cb,
                overlay_graphics_callback overlay_graphics_cb,
                nv_msgbroker_subscribe_cb_t msg_broker_subscribe_cb);

/**
 * @brief 暂停视频分析流水线。
 *
 * @param appCtx 应用程序上下文指针。
 * @return gboolean 暂停成功返回 TRUE，否则返回 FALSE。
 */
gboolean pause_pipeline(AppCtx *appCtx);

/**
 * @brief 恢复已暂停的视频分析流水线。
 *
 * @param appCtx 应用程序上下文指针。
 * @return gboolean 恢复成功返回 TRUE，否则返回 FALSE。
 */
gboolean resume_pipeline(AppCtx *appCtx);

/**
 * @brief 在流水线中进行定位跳转。
 *
 * @param appCtx 应用程序上下文指针。
 * @param milliseconds 跳转目标时间位置（毫秒）。
 * @param seek_is_relative 若为 TRUE 则为相对跳转，否则为绝对定位。
 * @return gboolean 跳转成功返回 TRUE，否则返回 FALSE。
 */
gboolean seek_pipeline(AppCtx *appCtx, glong milliseconds,
                       gboolean seek_is_relative);

/**
 * @brief 切换边界框文本显示状态。
 *
 * @param appCtx 应用程序上下文指针。
 */
void toggle_show_bbox_text(AppCtx *appCtx);

/**
 * @brief 销毁视频分析流水线并释放相关资源。
 *
 * @param appCtx 应用程序上下文指针。
 */
void destroy_pipeline(AppCtx *appCtx);

/**
 * @brief 重新启动视频分析流水线。
 *
 * 先销毁现有流水线，然后根据配置重新创建。
 *
 * @param appCtx 应用程序上下文指针。
 */
void restart_pipeline(AppCtx *appCtx);

/**
 * @brief 从配置文件读取属性配置。
 *
 * @param config [IN] 指向 NvDsConfig 的指针，用于存储解析的配置。
 * @param cfg_file_path [IN] 配置文件路径。
 * @return gboolean 解析成功返回 TRUE，否则返回 FALSE。
 */
gboolean parse_config_file(NvDsConfig *config, gchar *cfg_file_path);

/**
 * @brief 从 YML 配置文件读取属性配置。
 *
 * @param config [IN] 指向 NvDsConfig 的指针，用于存储解析的配置。
 * @param cfg_file_path [IN] YML 配置文件路径。
 * @return gboolean 解析成功返回 TRUE，否则返回 FALSE。
 */
gboolean parse_config_file_yaml(NvDsConfig *config, gchar *cfg_file_path);

/**
 * @brief 获取指定源 ID 对应的传感器信息。
 *
 * 通过 nvmultiurisrcbin REST API 添加流时获取 NvDsSensorInfo。
 *
 * @param appCtx [IN/OUT] 应用程序上下文，提供配置信息并维护流水线资源。
 * @param source_id [IN] NvDsFrameMeta 中的唯一源 ID。
 * @return [transfer-floating] 指向 NvDsSensorInfo 的指针。
 *         注意：返回指针仅在流被移除前有效。
 */
NvDsSensorInfo *get_sensor_info(AppCtx *appCtx, guint source_id);

#ifdef __cplusplus
}
#endif

#endif
