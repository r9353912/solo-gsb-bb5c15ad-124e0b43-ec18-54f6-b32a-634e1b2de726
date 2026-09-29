#!/usr/bin/env python3
"""定时策略切换时间边界调用示例（仅依赖 Python 标准库）。

演示内容（对应本轮修复）：
  1. HMAC-SHA256 签名调用管理写接口 PUT /policies、POST /policy-changes；
  2. 登记一条生效时刻为**整秒**（effective_at 文本形如 ...T12:00:00Z）的变更；
  3. 在生效点所在同一秒内、带微秒的时刻调用 POST /decide：
     变更必须已在本次余额计算前应用（capacity 100 -> 50）；
  4. 校验变更只应用一次、版本审计恰好一条 scheduled 记录；到期后取消返回 409；
  5. 预占 expire_at 落在整秒时，同秒带微秒的确认请求按到期优先转 expired 并返还。

用法：
    # 终端 A：启动服务（Compose 或本地均可）
    docker compose up --build
    #   或：ADMIN_KEY_ID=admin-key-1 ADMIN_HMAC_SECRET=change-me-please-use-openssl-rand-hex-32 \
    #       uvicorn app.main:app --port 8000

    # 终端 B：运行示例（默认指向 http://localhost:8000）
    python3 examples/policy_change_time_boundary.py
    BASE=http://localhost:8000 \\
    ADMIN_KEY_ID=admin-key-1 \\
    ADMIN_HMAC_SECRET=change-me-please-use-openssl-rand-hex-32 \\
        python3 examples/policy_change_time_boundary.py

时间边界约定：
    effective_at <= now 即到期，恰好到期也生效；未到期（哪怕只差 1 微秒）不提前
    应用。整秒（...Z）与带 6 位微秒（... .ffffffZ）的存量记录都按真实 UTC 判定。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

BASE = os.environ.get("BASE", "http://localhost:8000").rstrip("/")
KEY_ID = os.environ.get("ADMIN_KEY_ID", "admin-key-1")
SECRET = os.environ.get(
    "ADMIN_HMAC_SECRET", "change-me-please-use-openssl-rand-hex-32"
)
SUFFIX = str(time.time_ns())  # 让示例可反复运行：所有幂等键唯一


def call(method: str, path: str, body: dict | None = None, *, signed: bool = False):
    """发起一次 HTTP 调用；signed=True 时按管理写接口规则加 HMAC 头。"""
    raw = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json"}
    if signed:
        ts = str(int(time.time()))
        nonce = f"example-{time.time_ns()}"
        body_hash = hashlib.sha256(raw).hexdigest()
        signing = "\n".join([method, path, body_hash, ts, nonce]).encode()
        signature = hmac.new(
            SECRET.encode(), signing, hashlib.sha256
        ).hexdigest()
        headers.update(
            {
                "X-Key-Id": KEY_ID,
                "X-Timestamp": ts,
                "X-Nonce": nonce,
                "X-Signature": signature,
            }
        )
    req = urllib.request.Request(
        BASE + path, data=raw if raw else None, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def expect_ok(status: int, body, what: str) -> dict:
    if not 200 <= status < 300:
        sys.exit(f"[FAIL] {what} -> HTTP {status}: {body}")
    print(f"[ok]  {what} -> HTTP {status}")
    return body


def main() -> None:
    # 0) 健康检查
    with urllib.request.urlopen(BASE + "/health") as resp:
        assert resp.status == 200
    print(f"[ok]  GET /health（服务：{BASE}）")

    # 1) 配置全局策略：容量 100、速率 0（管理写接口，需签名）
    expect_ok(
        *call(
            "PUT",
            "/policies",
            {"scope": "global", "capacity": 100, "refill_rate": 0},
            signed=True,
        ),
        "PUT /policies（容量 100/速率 0）",
    )

    # 2) 登记 ~1.5 秒后、生效时刻截断到整秒的定时切换（容量 100 -> 50）
    eff = (datetime.now(timezone.utc) + timedelta(seconds=1, milliseconds=500)).replace(
        microsecond=0
    )
    eff_iso = eff.isoformat().replace("+00:00", "Z")
    change_id = f"ex-chg-{SUFFIX}"
    chg = expect_ok(
        *call(
            "POST",
            "/policy-changes",
            {
                "change_id": change_id,
                "scope": "global",
                "capacity": 50,
                "refill_rate": 0,
                "effective_at": eff_iso,
            },
            signed=True,
        ),
        f"POST /policy-changes（effective_at={eff_iso}，整秒）",
    )
    assert "." not in chg["effective_at"], chg["effective_at"]
    print(f"      库存 effective_at 文本：{chg['effective_at']}（无小数秒）")

    # 3) 等到生效点之后约 50ms（裁决时刻几乎必然落在同秒内且带微秒）
    wait = (eff - datetime.now(timezone.utc)).total_seconds() + 0.05
    time.sleep(max(0.0, wait))
    decision = expect_ok(
        *call(
            "POST",
            "/decide",
            {"request_id": f"ex-dec-{SUFFIX}", "cost": 1},
        ),
        "POST /decide（生效点之后同秒内裁决）",
    )
    cap = decision["buckets"][0]["capacity"]
    print(
        f"      decided_at={decision['decided_at']}，命中桶 capacity={cap}"
    )
    if cap != 50.0:
        sys.exit(f"[FAIL] 到期变更未在余额计算前应用：capacity={cap}")
    print("[ok]  到期变更已先应用，裁决看到的是新策略容量 50")

    # 4) 只应用一次 + 恰好一条 scheduled 版本
    status, got = call("GET", f"/policy-changes/{change_id}")
    assert status == 200 and got["status"] == "applied" and got["applied_at"]
    with urllib.request.urlopen(BASE + "/policies/history?scope=global") as resp:
        versions = json.loads(resp.read())
    scheduled = [v for v in versions if v["source"] == "scheduled"
                 and v["change_id"] == change_id]
    assert len(scheduled) == 1, versions
    print(f"[ok]  变更 applied（applied_at={got['applied_at']}），"
          f"scheduled 版本恰好 1 条（version={scheduled[0]['version']}）")

    # 5) 已到期再取消 -> 409（错误语义保持）
    status, body = call("DELETE", f"/policy-changes/{change_id}", signed=True)
    assert status == 409, (status, body)
    print(f"[ok]  DELETE 已到期变更 -> HTTP {status}（到期不可取消）")

    # 6) 预占：到期后（哪怕只过几十毫秒、仍在同一秒内）确认 -> 到期优先 expired
    reservation_id = f"ex-rsv-{SUFFIX}"
    rsv = expect_ok(
        *call(
            "POST",
            "/reservations",
            {"reservation_id": reservation_id, "cost": 5, "ttl_seconds": 1},
        ),
        "POST /reservations（预占 5，1 秒后到期）",
    )
    expire = datetime.fromisoformat(rsv["expire_at"].replace("Z", "+00:00"))
    wait = (expire - datetime.now(timezone.utc)).total_seconds() + 0.05
    time.sleep(max(0.0, wait))
    status, body = call("POST", f"/reservations/{reservation_id}/confirm")
    assert status == 409, (status, body)
    status, got = call("GET", f"/reservations/{reservation_id}")
    assert status == 200 and got["status"] == "expired"
    print(f"[ok]  expire_at={rsv['expire_at']} 之后确认 -> 409，"
          f"状态 expired（到期优先，已返还）")

    print("\n全部示例断言通过：整秒/小数秒 UTC 时间边界按真实先后生效。")


if __name__ == "__main__":
    main()
