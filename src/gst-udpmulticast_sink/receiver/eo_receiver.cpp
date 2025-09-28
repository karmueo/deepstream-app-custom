#include "eo_receiver.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>

EOReceiver::EOReceiver(const std::string& mcastIp, uint16_t port, const std::string& localIf)
    : mcastIp_(mcastIp), port_(port), localIf_(localIf) {}

EOReceiver::~EOReceiver() { stop(); }

bool EOReceiver::start() {
    if (running_) return true;

    sockfd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd_ < 0) {
        std::cerr << "EOReceiver: socket create failed: " << strerror(errno) << std::endl;
        return false;
    }

    int reuse = 1;
    if (setsockopt(sockfd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        std::cerr << "EOReceiver: setsockopt SO_REUSEADDR failed: " << strerror(errno) << std::endl;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port_);
    if (bind(sockfd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "EOReceiver: bind failed: " << strerror(errno) << std::endl;
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    ip_mreq mreq{};
    mreq.imr_multiaddr.s_addr = inet_addr(mcastIp_.c_str());
    if (!localIf_.empty()) {
        mreq.imr_interface.s_addr = inet_addr(localIf_.c_str());
    } else {
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
    }

    if (setsockopt(sockfd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        std::cerr << "EOReceiver: join multicast failed: " << strerror(errno) << std::endl;
        ::close(sockfd_);
        sockfd_ = -1;
        return false;
    }

    running_ = true;
    th_ = std::thread(&EOReceiver::recvLoop, this);
    return true;
}

void EOReceiver::stop() {
    if (!running_) return;
    running_ = false;
    if (sockfd_ >= 0) {
        ::shutdown(sockfd_, SHUT_RDWR);
    }
    if (th_.joinable()) th_.join();
    if (sockfd_ >= 0) {
        ::close(sockfd_);
        sockfd_ = -1;
    }
}

void EOReceiver::recvLoop() {
    constexpr size_t BUF_SIZE = 64 * 1024; // 足够容纳当前 JSON 报文
    std::vector<uint8_t> buf(BUF_SIZE);

    while (running_) {
        ssize_t n = ::recv(sockfd_, buf.data(), buf.size(), 0);
        if (n <= 0) {
            if (!running_) break;
            if (n < 0 && (errno == EINTR)) continue;
            if (n < 0) {
                std::cerr << "EOReceiver: recv error: " << strerror(errno) << std::endl;
            }
            continue;
        }

        MessageHeader header{};
        std::vector<EOTargetInfo> targets;
        if (EOProtocolParser::ParseEOTargetMessage(buf.data(), static_cast<size_t>(n), header, targets)) {
            if (callback_) {
                callback_(header, targets);
            } else {
                std::cout << "Received EO Target Message: count=" << targets.size() << std::endl;
                for (const auto& t : targets) {
                    std::cout << "  ID=" << t.targetID
                              << " class=" << static_cast<int>(t.targetClass)
                              << " conf=" << t.targetConfidence
                              << " offset(h,v)=" << t.offsetHorizontal << "," << t.offsetVertical
                              << " rect=" << t.targetRect << std::endl;
                }
            }
        } else {
            std::cerr << "EOReceiver: parse failed (size=" << n << ")" << std::endl;
        }
    }
}
