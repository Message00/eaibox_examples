#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

功能
- master 进程：监听一个 TCP 端口，接收 worker 的上线/心跳，维护节点列表，实时刷新终端界面
- worker 进程：主动连接 master，上报设备信息和 IP，定期心跳；断线自动重连

协议
- 文本行分隔的 JSON（NDJSON）。每条消息一行，UTF-8。
- worker -> master：
  {
    "type": "hello" | "heartbeat",     # 首条必须是 hello
    "id": "<stable_id>",
    "hostname": "...",
    "os": "...",
    "arch": "...",
    "python": "...",
    "ip": "<best_local_ip>",
    "extra": {"username": "...", "pid": 1234}
  }
- master -> worker：当前无主动下发，预留

运行示例
  # 启动 master（监听 0.0.0.0:9009）
  python netwatch.py master --host 0.0.0.0 --port 9009

  # 启动 worker（连接到 192.168.1.10:9009）
  python netwatch.py worker --master 192.168.1.10 --port 9009 --interval 5

依赖：仅标准库（asyncio, json, argparse, platform, socket, uuid, time, signal, shutil）
"""

from __future__ import annotations
import asyncio
import json
import argparse
import platform
import socket
import uuid
import time
import os
import signal
import sys
import shutil
from dataclasses import dataclass, field
from typing import Dict, Optional

# ----------------------------
# 公共工具
# ----------------------------

def best_local_ip() -> str:
    """尽可能拿到对外可路由的本机 IP（不依赖 DNS）。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "0.0.0.0"


def stable_machine_id() -> str:
    """通过 MAC + hostname 派生一个稳定 ID（避免裸露真实 MAC）。"""
    mac = uuid.getnode()  # 可能是随机或真实 MAC
    host = platform.node()
    # 简单混淆：哈希后取 12 字符
    import hashlib
    h = hashlib.sha256(f"{mac}-{host}".encode()).hexdigest()[:12]
    return h


# ----------------------------
# Master 实现
# ----------------------------

@dataclass
class Node:
    id: str
    hostname: str
    os: str
    arch: str
    python: str
    ip: str
    extra: dict = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    peer: str = ""  # 远端套接字地址


class Master:
    def __init__(self, host: str, port: int, ui_refresh: float = 1.0):
        self.host = host
        self.port = port
        self.nodes: Dict[str, Node] = {}
        self._server: Optional[asyncio.base_events.Server] = None
        self.ui_refresh = ui_refresh
        self._shutdown = asyncio.Event()

    async def start(self):
        self._server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = ", ".join(str(sock.getsockname()) for sock in self._server.sockets)
        print(f"[master] listening on {addr}")
        ui = asyncio.create_task(self.ui_loop())
        async with self._server:
            await self._shutdown.wait()
        ui.cancel()

    async def stop(self):
        self._shutdown.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info('peername')
        peer_s = f"{peer[0]}:{peer[1]}" if peer else "?"
        # 期望首条为 hello
        node_id: Optional[str] = None
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not line:
                writer.close()
                return
            msg = json.loads(line.decode('utf-8').strip())
            if msg.get("type") != "hello":
                writer.close()
                return
            node_id = str(msg.get("id"))
            node = Node(
                id=node_id,
                hostname=str(msg.get("hostname", "?")),
                os=str(msg.get("os", "?")),
                arch=str(msg.get("arch", "?")),
                python=str(msg.get("python", "?")),
                ip=str(msg.get("ip", "?")),
                extra=msg.get("extra", {}),
                peer=peer_s,
            )
            node.last_seen = time.time()
            self.nodes[node_id] = node
            # 心跳循环
            while not reader.at_eof():
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 超时当作断开
                    break
                if not line:
                    break
                try:
                    hb = json.loads(line.decode('utf-8').strip())
                    if hb.get("type") == "heartbeat":
                        n = self.nodes.get(node_id)
                        if n:
                            # 更新可变字段
                            n.ip = str(hb.get("ip", n.ip))
                            n.extra = hb.get("extra", n.extra)
                            n.last_seen = time.time()
                except Exception:
                    # 略过坏消息
                    continue
        except Exception:
            pass
        finally:
            if node_id and node_id in self.nodes:
                # 先标记为离线，保留 10 秒用于 UI 呈现“刚离线”状态
                self.nodes[node_id].last_seen = time.time() - 61
                # 真正删除延迟执行
                asyncio.create_task(self._delayed_remove(node_id))
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _delayed_remove(self, node_id: str, delay: float = 10.0):
        await asyncio.sleep(delay)
        self.nodes.pop(node_id, None)

    async def ui_loop(self):
        """每 ui_refresh 秒刷新一次终端表格。"""
        while True:
            self.render_ui()
            await asyncio.sleep(self.ui_refresh)

    def render_ui(self):
        cols = shutil.get_terminal_size((100, 20)).columns
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"NetWatch Master @ {self.host}:{self.port}  (nodes: {len(self.nodes)})")
        print("=" * min(cols, 120))
        headers = [
            ("ID", 12),
            ("Hostname", 18),
            ("OS", 16),
            ("Arch", 6),
            ("Python", 8),
            ("IP", 16),
            ("Peer", 21),
            ("Seen", 10),
        ]
        def fmt_cell(s, w):
            s = str(s)
            return s[:w-1] + '…' if len(s) > w else s.ljust(w)
        header_line = "  ".join(fmt_cell(h, w) for h, w in headers)
        print(header_line)
        print("-" * min(cols, 120))
        now = time.time()
        for n in sorted(self.nodes.values(), key=lambda x: (now - x.last_seen)):
            age = now - n.last_seen
            if age < 15:
                seen = "online"
            elif age < 60:
                seen = f"{int(age)}s"
            else:
                seen = "offline"
            row = [
                (n.id, 12),
                (n.hostname, 18),
                (n.os, 16),
                (n.arch, 6),
                (n.python, 8),
                (n.ip, 16),
                (n.peer, 21),
                (seen, 10),
            ]
            print("  ".join(fmt_cell(v, w) for v, w in row))
        print("\nHints: Ctrl+C 退出；节点断开后保留 10 秒用于观察；‘Seen’ 列展示在线/秒数/离线。")


# ----------------------------
# Worker 实现
# ----------------------------

class Worker:
    def __init__(self, master_host: str, port: int, interval: float = 5.0, name: Optional[str] = None):
        self.master_host = master_host
        self.port = port
        self.interval = interval
        self.name = name
        self._shutdown = asyncio.Event()

    def device_info(self):
        return {
            "id": self.name or stable_machine_id(),
            "hostname": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "arch": platform.machine(),
            "python": platform.python_version(),
            "ip": best_local_ip(),
            "extra": {
                "username": os.getenv("USER") or os.getenv("USERNAME") or "",
                "pid": os.getpid(),
            },
        }

    async def start(self):
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                await self.run_once()
                backoff = 1.0  # 正常返回（被 stop）
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[worker] connection error: {e}. retrying in {int(backoff)}s...")
                await asyncio.wait([self._shutdown.wait()], timeout=backoff)
                backoff = min(backoff * 2, 30)

    async def stop(self):
        self._shutdown.set()

    async def run_once(self):
        reader, writer = await asyncio.open_connection(self.master_host, self.port)
        print(f"[worker] connected to {self.master_host}:{self.port}")
        # hello
        hello = {"type": "hello", **self.device_info()}
        writer.write((json.dumps(hello) + "\n").encode("utf-8"))
        await writer.drain()
        try:
            # 心跳循环
            while not self._shutdown.is_set():
                hb = {"type": "heartbeat", **self.device_info()}
                writer.write((json.dumps(hb) + "\n").encode("utf-8"))
                await writer.drain()
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=self.interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


# ----------------------------
# CLI
# ----------------------------

def setup_signal_handlers(stop_coro):
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(stop_coro()))
        except NotImplementedError:
            # Windows 可能不支持 add_signal_handler
            pass


def parse_args():
    p = argparse.ArgumentParser(description="NetWatch: 轻量网络节点监控")
    sub = p.add_subparsers(dest="role", required=True)

    mp = sub.add_parser("master", help="运行 master 监听服务")
    mp.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    mp.add_argument("--port", type=int, default=9009, help="监听端口，默认 9009")
    mp.add_argument("--ui-refresh", type=float, default=1.0, help="UI 刷新间隔秒，默认 1.0")

    wp = sub.add_parser("worker", help="运行 worker 连接到 master")
    wp.add_argument("--master", required=True, help="master 主机名或 IP")
    wp.add_argument("--port", type=int, default=9009, help="master 端口，默认 9009")
    wp.add_argument("--interval", type=float, default=5.0, help="心跳秒数，默认 5 秒")
    wp.add_argument("--name", help="自定义节点 ID（默认基于机器信息派生）")

    return p.parse_args()


async def main():
    args = parse_args()
    if args.role == "master":
        m = Master(args.host, args.port, ui_refresh=args.ui_refresh)
        setup_signal_handlers(m.stop)
        await m.start()
    elif args.role == "worker":
        w = Worker(args.master, args.port, interval=args.interval, name=args.name)
        setup_signal_handlers(w.stop)
        await w.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
