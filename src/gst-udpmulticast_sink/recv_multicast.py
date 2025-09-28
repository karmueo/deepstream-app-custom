#!/usr/bin/env python3
import argparse
import socket
import struct
import sys
import time
from datetime import datetime

# C struct (packed) layout from gstudpmulticast_sink.h  (#pragma pack(1))
# struct _SendData {
#   float left, top, width, height;   // BboxInfo
#   int   class_id;                   // gint
#   unsigned long long object_id;     // guint64 (may be 0 if not filled)
#   float confidence;                 // gfloat
#   unsigned long long ntp_timestamp; // guint64
#   unsigned int source_id;           // guint
#   int   detect_class_id;            // gint
#   int   classify_class_id;          // gint
#   float classify_confidence;        // gfloat
# };
# Little-endian machine order was sent directly via sendto().
STRUCT_FMT = '<ffffiQfQIii f'.replace(' ', '')  # added detect_class_id, classify_class_id, classify_confidence
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)


def parse_args():
    p = argparse.ArgumentParser(description='Receive and decode DeepStream _udpmulticast_sink multicast packets.')
    p.add_argument('--group', default='239.255.10.10', help='Multicast group IP')
    p.add_argument('--port', type=int, default=6000, help='Multicast UDP port')
    p.add_argument('--iface', default='0.0.0.0', help='Local interface IP to bind / join (0.0.0.0 for default)')
    p.add_argument('--hex', action='store_true', help='Dump raw payload in hex')
    p.add_argument('--quiet', action='store_true', help='Minimal output (one line per packet)')
    return p.parse_args()


def join_multicast(sock: socket.socket, group: str, iface: str):
    # IP_ADD_MEMBERSHIP expects struct ip_mreq: imr_multiaddr (group), imr_interface (iface)
    mreq = struct.pack('=4s4s', socket.inet_aton(group), socket.inet_aton(iface))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    # Optional: allow multiple receivers
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # (Linux) also set loopback so local sender can see packets if needed
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    except OSError:
        pass


def decode_packet(data: bytes):
    if len(data) < STRUCT_SIZE:
        raise ValueError(f'Packet too small ({len(data)} bytes), expected {STRUCT_SIZE}')
    left, top, width, height, class_id, object_id, confidence, ntp_ts, source_id, detect_class_id, classify_class_id, classify_confidence = struct.unpack_from(STRUCT_FMT, data)
    return {
        'bbox': {
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'area': width * height,
        },
        'class_id': class_id,
        'object_id': object_id,
        'confidence': confidence,
        'ntp_timestamp': ntp_ts,
        'source_id': source_id,
        'detect_class_id': detect_class_id,
        'classify_class_id': classify_class_id,
        'classify_confidence': classify_confidence,
        'raw_len': len(data),
    }


def format_ntp(ts: int):
    # If this is actually a unix timestamp in ns/us/ms adapt heuristically.
    # We'll attempt a few simple guesses.
    if ts == 0:
        return '0'
    now = time.time()
    # Heuristics: choose scale that brings into plausible date range.
    for scale, label in [(1e0,'s'), (1e3,'ms'), (1e6,'us'), (1e9,'ns')]:
        t = ts / scale
        if 946684800 <= t <= now + 86400:  # between year 2000 and near future
            return datetime.fromtimestamp(t).isoformat() + f' (~{label})'
    return str(ts)


def main():
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Bind before membership (BSD allows either order; Linux usually OK either way)
    try:
        sock.bind((args.iface, args.port))  # listen on all for the given port
    except OSError as e:
        print(f'Bind failed on port {args.port}: {e}', file=sys.stderr)
        sys.exit(1)
    join_multicast(sock, args.group, args.iface)
    print(f'Listening on multicast {args.group}:{args.port} iface={args.iface} expecting {STRUCT_SIZE} bytes per packet')
    while True:
        data, addr = sock.recvfrom(2048)
        recv_time = time.time()
        try:
            decoded = decode_packet(data)
        except Exception as e:
            print(f'[WARN] {addr} len={len(data)} decode error: {e}')
            if args.hex:
                print(data.hex())
            continue
        if args.quiet:
            print(f"ts={recv_time:.6f} src={addr[0]}:{addr[1]} det={decoded['detect_class_id']} cls={decoded['classify_class_id']} dconf={decoded['confidence']:.3f} cconf={decoded['classify_confidence']:.3f} bbox=({decoded['bbox']['left']:.1f},{decoded['bbox']['top']:.1f},{decoded['bbox']['width']:.1f},{decoded['bbox']['height']:.1f}) src_id={decoded['source_id']}")
            continue
        print('-' * 60)
        print(f'Received from {addr[0]}:{addr[1]} bytes={decoded["raw_len"]}')
        print(f' Local recv time: {datetime.fromtimestamp(recv_time).isoformat()}')
        print(f" Detection Class ID (legacy/class_id): {decoded['class_id']}")
        print(f" Detect Class ID (explicit): {decoded['detect_class_id']}")
        print(f" Classification ID: {decoded['classify_class_id']}")
        print(f" Classification Confidence: {decoded['classify_confidence']:.4f}")
        print(f" Object ID: {decoded['object_id']}")
        print(f" Detection Confidence: {decoded['confidence']:.4f}")
        print(f" Source ID: {decoded['source_id']}")
        print(f" NTP Timestamp: {decoded['ntp_timestamp']} -> {format_ntp(decoded['ntp_timestamp'])}")
        bbox = decoded['bbox']
        print(f" BBox: left={bbox['left']:.1f} top={bbox['top']:.1f} w={bbox['width']:.1f} h={bbox['height']:.1f} area={bbox['area']:.1f}")
        if args.hex:
            print(' Raw Hex:', data[:STRUCT_SIZE].hex())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit.')
