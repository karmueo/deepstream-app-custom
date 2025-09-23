#ifndef EOPROTOCOLPARSER_H
#define EOPROTOCOLPARSER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// 前向声明 Json 相关类
namespace Json
{
class Value;
class StreamWriter;
class StreamWriterBuilder;
class CharReader;
class CharReaderBuilder;
} // namespace Json

// 系统类型定义
enum class SystemType : uint8_t
{
    S_BAND = 0,   // 0-S
    KUX_BAND = 1, // 1-Ku/X
    EO = 2,       // 2-光电
    FUSION = 3,   // 3-融合
    UAV = 4       // 4-无人机
};

// 消息类型定义
enum class MessageType : uint16_t
{
    CONTROL = 0,    // 0:控制
    FEEDBACK = 1,   // 1:回馈
    QUERY = 2,      // 2:查询
    DATA_STREAM = 3 // 3:数据流
};

// 报文类型定义
enum class BodyType : uint16_t
{
    JSON = 0,  // 0:json
    BINARY = 1 // 1:二进制
};

// 报文ID定义
enum class MessageID : uint16_t
{
    SYSTEM_PARAM_QUERY = 0x7101,       // 系统参数查询(指令)
    SYSTEM_PARAM = 0x7102,             // 系统参数
    BIT_STATUS_QUERY = 0x7101,         // 比特状态查询(指令)
    BIT_STATUS = 0x7103,               // 比特状态
    GUIDANCE_INFO = 0x7104,            // 引导信息
    TARGET_INFO = 0x7105,              // 目标信息 0x7105
    TRACK_CONTROL = 0x7106,            // 跟踪控制
    TRACK_CONTROL_FEEDBACK = 0x7106,   // 跟踪控制反馈
    SERVO_CONTROL = 0x7107,            // 伺服控制
    SERVO_CONTROL_FEEDBACK = 0x7107,   // 伺服控制反馈
    VISIBLE_LIGHT_CONTROL = 0x7108,    // 可见光控制
    VISIBLE_LIGHT_CONTROL_FB = 0x7108, // 可见光控制反馈
    INFRARED_CONTROL = 0x7109,         // 红外控制
    INFRARED_CONTROL_FB = 0x7109,      // 红外控制反馈
    RANGING = 0x7110,                  // 测距
    RANGING_FEEDBACK = 0x7110          // 测距反馈
};

// 目标状态定义
enum class TargetStatus : int
{
    NORMAL = 1,      // 1正常
    LOST = 0,        // 0丢失
    EXTRAPOLATED = 2 // 2外推
};

// 跟踪模式定义
enum class TrackMode : int
{
    DETECTION_TRACK = 0,  // 0检测跟踪
    RECOGNITION_TRACK = 1 // 1识别跟踪
};

// 目标类型定义
enum class TargetClass : int
{
    UNKNOWN = 0,         // 0:不明
    BIRD_FLOCK = 1,      // 1:鸟群
    AIRBORNE_OBJECT = 2, // 2:空飘物
    AIRCRAFT = 3,        // 3:飞机
    CAR = 4,             // 4:汽车
    BIG_BIRD = 5,        // 5:大鸟
    SMALL_BIRD = 6,      // 6:小鸟
    PEDESTRIAN = 7,      // 7:行人
    CRUISE_MISSILE = 8,  // 8:巡航导弹
    UAV = 9,             // 9:无人机
    UNKNOWN_15 = 15      // 15:未知
};

// 设备类型定义
enum class DeviceType : int
{
    VISIBLE_LIGHT = 0,  // 0可见光
    THERMAL_IMAGING = 1 // 1热成像
};

// 报文头结构体
#pragma pack(push, 1)
struct MessageHeader
{
    uint16_t messageID;     // 报文ID (字0)
    uint16_t messageLength; // 报文长度 (字1)
    uint16_t sendCount;     // 报文发送计数 (字2)
    uint16_t msgType;       // 消息类型 (字3)

    // 发送标识 (字4)
    struct SenderInfo
    {
        uint8_t    stationID;  // 站号 (b0-b7)
        SystemType systemType; // 系统类型 (b8-b15)
    };
    union SenderUnion
    {
        uint16_t   senderWord;
        SenderInfo senderInfo;
    } sender;

    // 接收标识 (字5)
    struct ReceiverInfo
    {
        uint8_t    stationID;  // 站号 (b0-b7)
        SystemType systemType; // 系统类型 (b8-b15)
    };
    union ReceiverUnion
    {
        uint16_t     receiverWord;
        ReceiverInfo receiverInfo;
    } receiver;

    // 发送时间 (字6-9)
    struct TimeInfo
    {
        uint8_t  year;      // 年 (基年:2000) (字6 b0-b7)
        uint8_t  month;     // 月 (字6 b8-b15)
        uint8_t  day;       // 日 (字7 b0-b7)
        uint8_t  hour;      // 时 (字7 b8-b15)
        uint8_t  minute;    // 分 (字8 b0-b7)
        uint8_t  second;    // 秒 (字8 b8-b15)
        uint16_t subSecond; // 秒以下 (量化单位:25us) (字9)
    };
    union TimeUnion
    {
        uint64_t timeWord;
        TimeInfo timeInfo;
    } time;

    // 内部分系统 (字10)
    struct SubsystemInfo
    {
        uint8_t senderSubsystem;   // 发端,内部分系统 (b0-b7)
        uint8_t receiverSubsystem; // 收端,内部分系统 (b8-b15)
    };
    union SubsystemUnion
    {
        uint16_t      subsystemWord;
        SubsystemInfo subsystemInfo;
    } subsystem;

    uint16_t backup1; // 备份 (字11)
    uint16_t backup2; // 备份 (字12)
    uint16_t backup3; // 备份 (字13)

    // 报文类型和长度 (字14)
    struct BodyInfo
    {
        uint16_t bodyType : 4;    // 类型 (b0-b3)
        uint16_t bodyLength : 12; // 单报文体长度 (b4-b15)
    };
    union BodyUnion
    {
        uint16_t bodyInfoWord;
        BodyInfo bodyInfo;
    } body;

    uint16_t bodyCount; // 报文体数 (字15)
};
#pragma pack(pop)

// 光电目标信息结构体
struct EOTargetInfo
{
    int          year;             // 年
    int          month;            // 月
    int          day;              // 日
    int          hour;             // 时
    int          minute;           // 分
    int          second;           // 秒
    float        msec;             // 毫秒
    DeviceType   deviceType;       // 设备类型
    int          targetID;         // 引导批号
    TargetStatus targetStatus;     // 目标状态
    TrackMode    trackMode;        // 跟踪模式
    float        fovAngle;         // 视场
    double       longitude;        // 站址经度
    double       latitude;         // 站址纬度
    double       altitude;         // 站址海拔高度
    float        fovHorizontal;    // 视场中心水平角度
    float        fovVertical;      // 视场中心垂直角度
    float        enuAzimuth;       // 目标水平角
    float        enuElevation;     // 目标垂直角
    int          offsetHorizontal; // 水平脱靶量
    int          offsetVertical;   // 垂直脱靶量
    int          targetRect;       // 目标位置元素
    TargetClass  targetClass;      // 目标类型
    float        targetConfidence; // 目标置信度
    float        targetDistance;   // 目标距离
};

// 光电报文封装和解析类
class EOProtocolParser
{
  public:
    // 封装光电目标信息报文（多目标）
    static std::vector<uint8_t>
    PackEOTargetMessage(const std::vector<EOTargetInfo> &targetInfos,
                        uint16_t                          sendCount,
                        uint8_t                           senderStation,
                        SystemType                        senderSystem,
                        uint8_t                           receiverStation,
                        SystemType                        receiverSystem,
                        uint8_t                           senderSubsystem = 0,
                        uint8_t                           receiverSubsystem = 0);

    // 解析光电目标信息报文（多目标）
    static bool ParseEOTargetMessage(const uint8_t           *data,
                                     size_t                   length,
                                     MessageHeader           &header,
                                     std::vector<EOTargetInfo> &targetInfos);

    // 计算校验和
    static uint16_t CalculateChecksum(const uint8_t *data, size_t length);

    // 验证帧尾
    static bool VerifyFrameTail(const uint8_t *data, size_t length);

  private:
    // 填充报文头
    static void FillMessageHeader(MessageHeader &header,
                                  MessageID      messageID,
                                  uint16_t       bodyLength,
                                  uint16_t       sendCount,
                                  MessageType    msgType,
                                  uint8_t        senderStation,
                                  SystemType     senderSystem,
                                  uint8_t        receiverStation,
                                  SystemType     receiverSystem,
                                  const MessageHeader::TimeInfo &timeInfo,
                                  uint8_t senderSubsystem,
                                  uint8_t receiverSubsystem);

    // 从JSON解析目标信息
    static bool ParseTargetInfoFromJson(const Json::Value &json,
                                        EOTargetInfo      &targetInfo);

    // 生成目标信息的JSON
    static Json::Value CreateTargetInfoJson(const EOTargetInfo &targetInfo);

    // 获取当前时间信息
    static MessageHeader::TimeInfo GetCurrentTime();

    // JSON 写入器构建器
    static std::unique_ptr<Json::StreamWriterBuilder> GetWriterBuilder();

    // JSON 读取器构建器
    static std::unique_ptr<Json::CharReaderBuilder> GetReaderBuilder();
};

#endif // EOPROTOCOLPARSER_H