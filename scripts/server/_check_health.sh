#!/bin/bash
source ~/vllm-venv/bin/activate
export no_proxy="*"
python3 -c "
import requests
try:
    r = requests.get('http://127.0.0.1:8000/health', timeout=3)
    print('SERVICE: OK (HTTP', r.status_code, ')')
except Exception as e:
    print('SERVICE: DOWN -', e)
"
