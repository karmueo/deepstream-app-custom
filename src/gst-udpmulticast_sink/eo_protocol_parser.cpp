#include "eo_protocol_parser.h"
#include <arpa/inet.h>
#include <cstring>
#include <ctime>
#include <iostream>
#include <jsoncpp/json/reader.h>
#include <jsoncpp/json/value.h>
#include <jsoncpp/json/writer.h>
#include <memory>
#include <sstream>

// 帧尾同步字
const uint16_t FRAME_TAIL = 0x55AA;

// 字节序转换宏
#if defined(__linux__) || defined(__APPLE__)
#include <byteswap.h>
#define BSWAP_64(x) bswap_64(x)
#elif defined(_WIN32)
#include <stdlib.h>
#define BSWAP_64(x) _byteswap_uint64(x)
#else
#define BSWAP_64(x)                                                            \
    ((((x)&0xff00000000000000ull) >> 56) |                                     \
     (((x)&0x00ff000000000000ull) >> 40) |                                     \
     (((x)&0x0000ff0000000000ull) >> 24) |                                     \
     (((x)&0x000000ff00000000ull) >> 8) | (((x)&0x00000000ff000000ull) << 8) | \
     (((x)&0x0000000000ff0000ull) << 24) |                                     \
     (((x)&0x000000000000ff00ull) << 40) |                                     \
     (((x)&0x00000000000000ffull) << 56))
#endif

std::vector<uint8_t>
EOProtocolParser::PackEOTargetMessage(const std::vector<EOTargetInfo> &targetInfos,
                                      uint16_t                          sendCount,
                                      uint8_t                           senderStation,
                                      SystemType                        senderSystem,
                                      uint8_t                           receiverStation,
                                      SystemType                        receiverSystem,
                                      uint8_t                           senderSubsystem,
                                      uint8_t                           receiverSubsystem)
{
    // 如果没有目标信息，返回空消息
    if (targetInfos.empty()) {
        return std::vector<uint8_t>();
    }

    // 创建JSON报文体 - 包含多个目标
    Json::Value jsonBody;
    jsonBody["targets"] = Json::Value(Json::arrayValue);
    
    for (const auto& targetInfo : targetInfos) {
        Json::Value targetJson = CreateTargetInfoJson(targetInfo);
        jsonBody["targets"].append(targetJson);
    }
    
    auto        writerBuilder = GetWriterBuilder();
    std::string jsonStr = Json::writeString(*writerBuilder, jsonBody);

    // 计算报文总长度
    uint16_t bodyLength = static_cast<uint16_t>(jsonStr.size());
    uint16_t totalLength = sizeof(MessageHeader) + bodyLength +
                           2 * sizeof(uint16_t); // 头+体+校验和+帧尾

    // 准备报文缓冲区
    std::vector<uint8_t> message(totalLength);
    uint8_t             *data = message.data();

    // 填充报文头
    MessageHeader header;
    FillMessageHeader(header, MessageID::TARGET_INFO, bodyLength, sendCount,
                      MessageType::DATA_STREAM, senderStation, senderSystem,
                      receiverStation, receiverSystem, GetCurrentTime(),
                      senderSubsystem, receiverSubsystem);

    // 拷贝报文头
    std::memcpy(data, &header, sizeof(MessageHeader));
    data += sizeof(MessageHeader);

    // 拷贝JSON报文体
    std::memcpy(data, jsonStr.c_str(), bodyLength);
    data += bodyLength;

    // 计算并填充校验和
    uint16_t checksum =
        CalculateChecksum(message.data(), sizeof(MessageHeader) + bodyLength);
    *reinterpret_cast<uint16_t *>(data) = htons(checksum);
    data += sizeof(uint16_t);

    // 填充帧尾
    *reinterpret_cast<uint16_t *>(data) = htons(FRAME_TAIL);

    return message;
}

bool EOProtocolParser::ParseEOTargetMessage(const uint8_t           *data,
                                            size_t                   length,
                                            MessageHeader           &header,
                                            std::vector<EOTargetInfo> &targetInfos)
{
    if (length < sizeof(MessageHeader) + 2 * sizeof(uint16_t))
    {
        return false;
    }

    // 解析报文头
    std::memcpy(&header, data, sizeof(MessageHeader));

    // 字节序转换
    header.messageID = ntohs(header.messageID);
    header.messageLength = ntohs(header.messageLength);
    header.sendCount = ntohs(header.sendCount);
    header.msgType = ntohs(header.msgType);
    header.sender.senderWord = ntohs(header.sender.senderWord);
    header.receiver.receiverWord = ntohs(header.receiver.receiverWord);
    header.time.timeWord = BSWAP_64(header.time.timeWord); // 64位字节序转换
    header.subsystem.subsystemWord = ntohs(header.subsystem.subsystemWord);
    header.backup1 = ntohs(header.backup1);
    header.backup2 = ntohs(header.backup2);
    header.backup3 = ntohs(header.backup3);
    header.body.bodyInfoWord = ntohs(header.body.bodyInfoWord);
    header.bodyCount = ntohs(header.bodyCount);

    // 检查报文ID
    if (header.messageID != static_cast<uint16_t>(MessageID::TARGET_INFO))
    {
        return false;
    }

    // 验证校验和
    size_t dataLength = sizeof(MessageHeader) + header.body.bodyInfo.bodyLength;
    uint16_t expectedChecksum = CalculateChecksum(data, dataLength);
    uint16_t actualChecksum =
        ntohs(*reinterpret_cast<const uint16_t *>(data + dataLength));

    if (expectedChecksum != actualChecksum)
    {
        return false;
    }

    // 验证帧尾
    if (!VerifyFrameTail(data, length))
    {
        return false;
    }

    // 解析JSON报文体
    const char *jsonStr =
        reinterpret_cast<const char *>(data + sizeof(MessageHeader));
    auto        readerBuilder = GetReaderBuilder();
    Json::Value jsonBody;
    std::string errors;

    std::istringstream jsonStream(
        std::string(jsonStr, header.body.bodyInfo.bodyLength));
    if (!Json::parseFromStream(*readerBuilder, jsonStream, &jsonBody, &errors))
    {
        return false;
    }

    // 清空目标列表
    targetInfos.clear();

    // 检查是否是多目标格式（包含targets数组）
    if (jsonBody.isMember("targets") && jsonBody["targets"].isArray()) {
        // 多目标格式
        const Json::Value& targetsArray = jsonBody["targets"];
        for (const auto& targetJson : targetsArray) {
            EOTargetInfo targetInfo;
            if (ParseTargetInfoFromJson(targetJson, targetInfo)) {
                targetInfos.push_back(targetInfo);
            }
        }
    } else {
        // 单目标格式（向后兼容）
        EOTargetInfo targetInfo;
        if (ParseTargetInfoFromJson(jsonBody, targetInfo)) {
            targetInfos.push_back(targetInfo);
        }
    }

    return !targetInfos.empty();
}

uint16_t EOProtocolParser::CalculateChecksum(const uint8_t *data, size_t length)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < length; i += 2)
    {
        if (i + 1 < length)
        {
            sum += (data[i] << 8) | data[i + 1];
        }
        else
        {
            sum += data[i] << 8;
        }
    }

    while (sum >> 16)
    {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return static_cast<uint16_t>(~sum);
}

bool EOProtocolParser::VerifyFrameTail(const uint8_t *data, size_t length)
{
    if (length < 2 * sizeof(uint16_t))
    {
        return false;
    }

    const uint16_t *frameTail =
        reinterpret_cast<const uint16_t *>(data + length - sizeof(uint16_t));
    return ntohs(*frameTail) == FRAME_TAIL;
}

void EOProtocolParser::FillMessageHeader(
    MessageHeader                 &header,
    MessageID                      messageID,
    uint16_t                       bodyLength,
    uint16_t                       sendCount,
    MessageType                    msgType,
    uint8_t                        senderStation,
    SystemType                     senderSystem,
    uint8_t                        receiverStation,
    SystemType                     receiverSystem,
    const MessageHeader::TimeInfo &timeInfo,
    uint8_t                        senderSubsystem,
    uint8_t                        receiverSubsystem)
{
    header.messageID = htons(static_cast<uint16_t>(messageID));
    header.messageLength =
        htons(sizeof(MessageHeader) + bodyLength + 2 * sizeof(uint16_t));
    header.sendCount = htons(sendCount);
    header.msgType = htons(static_cast<uint16_t>(msgType));

    header.sender.senderInfo.stationID = senderStation;
    header.sender.senderInfo.systemType = senderSystem;
    header.sender.senderWord = htons(header.sender.senderWord);

    header.receiver.receiverInfo.stationID = receiverStation;
    header.receiver.receiverInfo.systemType = receiverSystem;
    header.receiver.receiverWord = htons(header.receiver.receiverWord);

    header.time.timeInfo = timeInfo;
    header.time.timeWord = BSWAP_64(header.time.timeWord);

    header.subsystem.subsystemInfo.senderSubsystem = senderSubsystem;
    header.subsystem.subsystemInfo.receiverSubsystem = receiverSubsystem;
    header.subsystem.subsystemWord = htons(header.subsystem.subsystemWord);

    header.backup1 = 0;
    header.backup2 = 0;
    header.backup3 = 0;

    header.body.bodyInfo.bodyType = static_cast<uint16_t>(BodyType::JSON);
    header.body.bodyInfo.bodyLength = bodyLength;
    header.body.bodyInfoWord = htons(header.body.bodyInfoWord);

    header.bodyCount = htons(1);
}

bool EOProtocolParser::ParseTargetInfoFromJson(const Json::Value &json,
                                               EOTargetInfo      &targetInfo)
{
    try
    {
        targetInfo.year = json["yr"].asInt();
        targetInfo.month = json["mo"].asInt();
        targetInfo.day = json["dy"].asInt();
        targetInfo.hour = json["h"].asInt();
        targetInfo.minute = json["min"].asInt();
        targetInfo.second = json["sec"].asInt();
        targetInfo.msec = json["msec"].asFloat();
        targetInfo.deviceType = static_cast<DeviceType>(json["dev_id"].asInt());
        targetInfo.targetID = json["tar_id"].asInt();
        targetInfo.targetStatus =
            static_cast<TargetStatus>(json["trk_stat"].asInt());
        targetInfo.trackMode = static_cast<TrackMode>(json["trk_mod"].asInt());
        targetInfo.fovAngle = json["fov_angle"].asFloat();
        targetInfo.longitude = json["lon"].asDouble();
        targetInfo.latitude = json["lat"].asDouble();
        targetInfo.altitude = json["alt"].asDouble();
        targetInfo.fovHorizontal = json["fov_h"].asFloat();
        targetInfo.fovVertical = json["fov_v"].asFloat();
        targetInfo.enuAzimuth = json["enu_a"].asFloat();
        targetInfo.enuElevation = json["enu_e"].asFloat();
        targetInfo.offsetHorizontal = json["offset_h"].asInt();
        targetInfo.offsetVertical = json["offset_v"].asInt();
        targetInfo.targetRect = json["tar_rect"].asInt();
        targetInfo.targetClass =
            static_cast<TargetClass>(json["tar_class"].asInt());
        targetInfo.targetConfidence = json["tar_dfid"].asFloat();
        targetInfo.targetDistance = json["tar_mg"].asFloat();

        return true;
    }
    catch (...)
    {
        return false;
    }
}

Json::Value
EOProtocolParser::CreateTargetInfoJson(const EOTargetInfo &targetInfo)
{
    Json::Value json;
    json["yr"] = targetInfo.year;
    json["mo"] = targetInfo.month;
    json["dy"] = targetInfo.day;
    json["h"] = targetInfo.hour;
    json["min"] = targetInfo.minute;
    json["sec"] = targetInfo.second;
    json["msec"] = targetInfo.msec;
    json["dev_id"] = static_cast<int>(targetInfo.deviceType);
    json["tar_id"] = targetInfo.targetID;
    json["trk_stat"] = static_cast<int>(targetInfo.targetStatus);
    json["trk_mod"] = static_cast<int>(targetInfo.trackMode);
    json["fov_angle"] = targetInfo.fovAngle;
    json["lon"] = targetInfo.longitude;
    json["lat"] = targetInfo.latitude;
    json["alt"] = targetInfo.altitude;
    json["fov_h"] = targetInfo.fovHorizontal;
    json["fov_v"] = targetInfo.fovVertical;
    json["enu_a"] = targetInfo.enuAzimuth;
    json["enu_e"] = targetInfo.enuElevation;
    json["offset_h"] = targetInfo.offsetHorizontal;
    json["offset_v"] = targetInfo.offsetVertical;
    json["tar_rect"] = targetInfo.targetRect;
    json["tar_class"] = static_cast<int>(targetInfo.targetClass);
    json["tar_dfid"] = targetInfo.targetConfidence;
    json["tar_mg"] = targetInfo.targetDistance;

    return json;
}

MessageHeader::TimeInfo EOProtocolParser::GetCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm    *tm = std::localtime(&now);

    MessageHeader::TimeInfo timeInfo;
    timeInfo.year = static_cast<uint8_t>(tm->tm_year % 100); // 基于2000年
    timeInfo.month = static_cast<uint8_t>(tm->tm_mon + 1);
    timeInfo.day = static_cast<uint8_t>(tm->tm_mday);
    timeInfo.hour = static_cast<uint8_t>(tm->tm_hour);
    timeInfo.minute = static_cast<uint8_t>(tm->tm_min);
    timeInfo.second = static_cast<uint8_t>(tm->tm_sec);
    timeInfo.subSecond = 0; // 可根据需要填充毫秒信息

    return timeInfo;
}

std::unique_ptr<Json::StreamWriterBuilder> EOProtocolParser::GetWriterBuilder()
{
    auto builder = std::make_unique<Json::StreamWriterBuilder>();
    builder->settings_["indentation"] = ""; // 紧凑格式，无缩进
    return builder;
}

std::unique_ptr<Json::CharReaderBuilder> EOProtocolParser::GetReaderBuilder()
{
    return std::make_unique<Json::CharReaderBuilder>();
}