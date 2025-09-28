#ifndef EO_RECEIVER_H
#define EO_RECEIVER_H

#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>

// 头文件位于上级目录，CMake 中通过 target_include_directories 添加当前列表目录即可找到
#include "eo_protocol_parser.h"

// 简单的 UDP 组播接收器, 接收 EO 多目标报文并解析打印
class EOReceiver {
public:
    using TargetCallback = std::function<void(const MessageHeader&, const std::vector<EOTargetInfo>&)>;

    EOReceiver(const std::string& mcastIp, uint16_t port, const std::string& localIf = "");
    ~EOReceiver();

    // 启动接收线程
    bool start();
    // 停止接收线程
    void stop();

    void setCallback(TargetCallback cb) { callback_ = std::move(cb); }

private:
    void recvLoop();

    std::string mcastIp_;
    uint16_t port_{};
    std::string localIf_;

    int sockfd_{-1};
    std::thread th_;
    std::atomic<bool> running_{false};

    TargetCallback callback_;
};

#endif // EO_RECEIVER_H
