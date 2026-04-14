#!/usr/bin/env python3
"""快速测试 streaming chat 请求"""
import asyncio
import json
import aiohttp

async def main():
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                print(f"Status: {resp.status}")
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        print(f"  >> {decoded}")
    except Exception as e:
        print(f"ERROR: {e}")

asyncio.run(main())
