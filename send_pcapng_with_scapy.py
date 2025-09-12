#!/usr/bin/env python3
"""
使用Scapy发送指定的pcapng文件
支持多播发送和自定义发送参数
"""

import argparse
import time
import sys
import os
import socket
import struct
from scapy.all import *

def send_pcapng_with_socket(pcap_file, delay=0.1, count=1, multicast_group='239.255.10.10', port=6000):
    """
    使用socket方式发送pcapng文件中的UDP多播数据包
    提取Raw payload并通过UDP socket发送
    """
    
    # 检查文件是否存在
    if not os.path.exists(pcap_file):
        print(f"错误: 文件 {pcap_file} 不存在")
        return False
    
    try:
        # 读取pcapng文件
        print(f"正在读取 {pcap_file}...")
        packets = rdpcap(pcap_file)
        print(f"成功读取 {len(packets)} 个数据包")
        
        # 提取UDP payload
        payloads = []
        for pkt in packets:
            if IP in pkt and UDP in pkt and Raw in pkt:
                # 只处理目标端口6000的多播UDP数据包
                if pkt[IP].dst.startswith('239.') and pkt[UDP].dport == 6000:
                    payloads.append(bytes(pkt[Raw].load))
        
        print(f"提取到 {len(payloads)} 个有效的多播UDP payload")
        if not payloads:
            print("错误: 没有找到有效的多播UDP数据包")
            return False
        
        # 创建UDP socket用于多播发送
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        
        # 设置多播TTL
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 64)
        
        print(f"\n开始发送数据包...")
        print(f"目标: {multicast_group}:{port}")
        print(f"延迟: {delay} 秒")
        print(f"发送次数: {count}")
        print(f"每轮发送包数: {len(payloads)}")
        print("=" * 50)
        
        total_sent = 0
        
        for round_num in range(count):
            print(f"\n第 {round_num + 1} 轮发送:")
            
            for i, payload in enumerate(payloads):
                try:
                    # 发送UDP数据包到多播组
                    sock.sendto(payload, (multicast_group, port))
                    total_sent += 1
                    print(f"  发送包 {i+1}/{len(payloads)}: {len(payload)} 字节")
                    
                    # 延迟
                    if delay > 0:
                        time.sleep(delay)
                        
                except Exception as e:
                    print(f"  发送包 {i+1} 失败: {e}")
            
            print(f"第 {round_num + 1} 轮发送完成 ({len(payloads)} 包)")
            
            # 如果还有下一轮，稍等一下
            if round_num < count - 1:
                time.sleep(1)
        
        sock.close()
        print(f"\n所有数据包发送完成! 总共发送: {total_sent} 包")
        return True
        
    except Exception as e:
        print(f"发送过程中出现错误: {e}")
        return False

def list_interfaces():
    """列出可用的网络接口"""
    print("可用的网络接口:")
    try:
        for iface in get_if_list():
            print(f"  {iface}")
    except Exception as e:
        print(f"无法获取接口列表: {e}")

def analyze_pcapng(pcap_file):
    """分析pcapng文件内容"""
    if not os.path.exists(pcap_file):
        print(f"错误: 文件 {pcap_file} 不存在")
        return
    
    try:
        packets = rdpcap(pcap_file)
        print(f"\n文件: {pcap_file}")
        print(f"总包数: {len(packets)}")
        print("=" * 50)
        
        # 统计协议分布
        protocol_stats = {}
        for pkt in packets:
            if IP in pkt:
                proto = pkt[IP].proto
                protocol_stats[proto] = protocol_stats.get(proto, 0) + 1
        
        print("协议分布:")
        for proto, count in protocol_stats.items():
            proto_name = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}.get(proto, f'Proto-{proto}')
            print(f"  {proto_name}: {count} 包")
        
        # 显示详细信息
        print("\n详细包信息:")
        for i, pkt in enumerate(packets):
            print(f"包 {i+1}:")
            print(f"  摘要: {pkt.summary()}")
            
            if IP in pkt:
                print(f"  源IP: {pkt[IP].src}")
                print(f"  目标IP: {pkt[IP].dst}")
                
            if UDP in pkt:
                print(f"  源端口: {pkt[UDP].sport}")
                print(f"  目标端口: {pkt[UDP].dport}")
                print(f"  负载长度: {len(pkt[UDP].payload)}")
                
            if TCP in pkt:
                print(f"  源端口: {pkt[TCP].sport}")
                print(f"  目标端口: {pkt[TCP].dport}")
                
            print()
            
    except Exception as e:
        print(f"分析文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='使用Scapy发送pcapng文件')
    parser.add_argument('pcap_file', nargs='?', help='要发送的pcapng文件路径')
    parser.add_argument('-i', '--interface', help='指定网络接口')
    parser.add_argument('-d', '--delay', type=float, default=0.1, help='包之间的延迟（秒），默认0.1')
    parser.add_argument('-c', '--count', type=int, default=1, help='发送次数，默认1')
    parser.add_argument('-m', '--multicast', help='多播组地址（如224.1.1.1）')
    parser.add_argument('-p', '--port', type=int, help='目标端口')
    parser.add_argument('--list-interfaces', action='store_true', help='列出可用的网络接口')
    parser.add_argument('--analyze', action='store_true', help='只分析文件，不发送')
    
    args = parser.parse_args()
    
    # 列出接口
    if args.list_interfaces:
        list_interfaces()
        return
    
    # 检查是否提供了文件路径
    if not args.pcap_file:
        parser.error("需要指定pcapng文件路径，除非使用 --list-interfaces")
    
    # 分析文件
    if args.analyze:
        analyze_pcapng(args.pcap_file)
        return
    
    # 检查是否为root用户（不再必需，因为使用socket发送）
    # if os.geteuid() != 0:
    #     print("警告: 发送原始数据包通常需要root权限")
    #     print("如果发送失败，请尝试使用 sudo 运行此脚本")
    
    # 发送数据包（使用socket方式）
    success = send_pcapng_with_socket(
        pcap_file=args.pcap_file,
        delay=args.delay,
        count=args.count,
        multicast_group=args.multicast or '239.255.10.10',  # 默认使用pcap中的多播组
        port=args.port or 6000  # 默认使用pcap中的端口
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()