"""FastAPI 入口：多级配额裁决服务。

启动后默认监听 8000 端口，交互式文档位于 /docs。
每个写请求使用独立连接 + BEGIN IMMEDIATE 事务，天然串行化，
并发裁决不会超发（见 app/database.py 与 app/quota.py）。

管理写接口（PUT /policies、POST /policies/rollback、POST /policy-changes、
DELETE /policy-changes/{change_id}、POST /route-transfers）需要 HMAC-SHA256
防重放签名（X-Key-Id/X-Timestamp/X-Nonce/X-Signature，见 app/auth.py）；
所有 GET、POST /decide、POST /decide-batches 裁决与撤销、
POST /policies/impact-preview 预演、POST /reservations 预占
（含确认/取消/结算/查询）与 POST /quota-plans 配额规划接口保持匿名可用。
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import quota
from .auth import AdminPrincipal, require_admin, reserve_nonce
from .database import connect, init_db, read_tx, write_tx
from .schemas import (
    BucketOut,
    DecideBatchIn,
    DecideBatchOut,
    DecideIn,
    DecisionOut,
    ImpactPreviewIn,
    ImpactPreviewOut,
    LowBalanceEventOut,
    PolicyChangeIn,
    PolicyChangeOut,
    PolicyIn,
    PolicyOut,
    PolicyRollbackIn,
    PolicyVersionOut,
    QuotaPlanIn,
    QuotaPlanOut,
    QuotaUsageOut,
    ReservationIn,
    ReservationOut,
    ReservationSettleIn,
    RevokeOut,
    RouteTransferIn,
    RouteTransferOut,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="多级配额裁决 API",
    description=(
        "全局 / 租户 / 路由三级令牌桶配额裁决。"
        "一次裁决在同一事务中补充并扣减所有命中桶，全部满足才扣减；"
        "request_id 幂等重放；支持撤销补回；"
        "策略可配置 low_balance_threshold，余额越线时产生 low_balance 事件；"
        "支持提交未来生效的策略定时切换，由裁决在同一事务内惰性应用；"
        "策略变更追加不可变版本记录，支持历史查询与同事务回滚。"
        "POST /decide-batches 在一个 BEGIN IMMEDIATE 事务内整批裁决 1～50 个请求："
        "同一时刻应用命中的到期切换，按数组顺序演算共享桶，全部获准才落库，"
        "批次按 batch_id 幂等存档。"
        "POST /policies/impact-preview 在只读快照上预演候选容量/速率对"
        "未来 0～3600 秒请求序列的影响，双侧逐条对比，不落库。"
        "POST /reservations 提供配额预占：全部命中桶充足才统一预扣并保存原命中桶，"
        "否则全部不扣；支持确认（仅 pending->confirmed，不再扣减）、"
        "取消（pending->cancelled，向原桶返还）、结算（未到期 pending->settled，"
        "按实际用量多退少补、补扣不足保持 pending 不扣款）"
        "与查询（到期惰性转 expired 并返还），"
        "预占 1～300 秒到期，所有余额写操作先在同一事务内完成到期返还。"
        "POST /quota-plans 在只读快照上为 1～18 个待执行请求规划配额内总收益最大的"
        "子集：同一时刻按现行策略补充各命中桶（未建桶按满额），不推进定时变更与"
        "预占状态，精确求最优（总收益最大、项数更多、升序 ID 字典序更小），全程只读。"
        "POST /route-transfers 在同一租户的两个不同路由桶之间移拨现有令牌："
        "独立前序写事务先应用两侧到期的定时切换、清扫并返还到期预占、把两桶补充到"
        "同一时刻（即使移拨冲突也已生效），随后只扣源桶、增目标桶（不扣全局/租户桶、"
        "不写裁决与低余额事件）；源不足或目标超容量返回 409；transfer_id 对完整载荷"
        "幂等重放。"
        "GET /quota-usage 在同一只读快照内按 UTC 左闭右开窗口（裁决时间或预占创建"
        "时间筛选）按 tenant_id、route_id 汇总已消耗量（获准未撤销裁决、confirmed/"
        "settled 预占）与未完成占用量（报表时刻尚未到期的 pending 预占；到期未清扫"
        "也按到期处理），不修改预占或桶。"
        "管理写接口（策略新增/更新、回滚、定时变更创建/取消、路由桶令牌移拨）需要"
        " HMAC-SHA256 防重放签名（X-Key-Id/X-Timestamp/X-Nonce/X-Signature）；"
        "GET、裁决（含批量）、预演、撤销、预占、配额规划与配额消耗核对接口无需认证。"
    ),
    version="1.12.0",
    lifespan=lifespan,
)


def get_conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def _sanitize_jsonable(obj: Any) -> Any:
    """把响应结构中的非有限浮点（NaN / Infinity）替换为安全字符串，
    保证 422 校验错误即使回显了 NaN 输入也能正常 JSON 序列化。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else f"<non-finite-float:{obj!s}>"
    if isinstance(obj, dict):
        return {k: _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_jsonable(v) for v in obj]
    return obj


def _parse_utc_window_param(name: str, raw: str) -> datetime:
    """把查询参数解析为显式带时区的 UTC 时刻。

    时间格式无效（无法按 ISO 8601 解析）或不带时区（无法可靠解释为 UTC）
    一律返回 422 校验错误。
    """
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            422,
            f"查询参数 {name} 不是合法的 ISO 8601 UTC 时间"
            "（例如 2026-09-27T12:00:00Z）",
        )
    if dt.tzinfo is None:
        raise HTTPException(
            422, f"查询参数 {name} 必须显式带时区（UTC，例如 ...T12:00:00Z）"
        )
    return dt.astimezone(timezone.utc)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 与 FastAPI 默认 422 响应结构一致（仅保留可序列化的键，剔除含异常对象的 ctx），
    # 另把回显输入中的非有限浮点替换为安全字符串，使 NaN / Infinity 输入的 422
    # 错误响应自身也能正常 JSON 序列化。
    detail = []
    for err in exc.errors():
        clean = {k: v for k, v in err.items() if k != "ctx"}
        if "input" in clean:
            clean["input"] = _sanitize_jsonable(clean["input"])
        detail.append(clean)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------- 策略


@app.get("/policies", response_model=list[PolicyOut], tags=["策略"])
def get_policies(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    return quota.list_policies(conn)


@app.put(
    "/policies",
    response_model=PolicyOut,
    tags=["策略"],
    responses={
        401: {"description": "缺少/无效的 HMAC 签名，或时间戳偏离超过 300 秒"},
        409: {"description": "nonce 在同一 key 下 300 秒内重用；或该策略存在 pending 变更"},
        503: {"description": "服务端未配置 ADMIN_KEY_ID / ADMIN_HMAC_SECRET"},
    },
)
def put_policy(
    body: PolicyIn,
    conn: sqlite3.Connection = Depends(get_conn),
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    # nonce 登记与业务写入在同一 BEGIN IMMEDIATE 事务：业务失败回滚时
    # nonce 不被占用；提交成功后 300 秒内同 key 同 nonce 一律 409
    with write_tx(conn):
        reserve_nonce(conn, admin.key_id, admin.nonce)
        return quota.upsert_policy(
            conn,
            scope=body.scope,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            capacity=body.capacity,
            refill_rate=body.refill_rate,
            low_balance_threshold=body.low_balance_threshold,
        )


@app.get(
    "/policies/history",
    response_model=list[PolicyVersionOut],
    tags=["策略版本审计"],
)
def get_policy_history(
    scope: str = Query(..., pattern="^(global|tenant|route)$"),
    tenant_id: str | None = Query(default=None),
    route_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    """查询单个策略的不可变版本历史，按版本号新到旧返回。只读。"""
    return quota.list_policy_versions(
        conn, scope=scope, tenant_id=tenant_id, route_id=route_id
    )


@app.post(
    "/policies/rollback",
    response_model=PolicyVersionOut,
    tags=["策略版本审计"],
    responses={
        401: {"description": "缺少/无效的 HMAC 签名，或时间戳偏离超过 300 秒"},
        404: {"description": "目标策略或版本不存在"},
        409: {"description": "nonce 重用；或该策略存在待生效（pending）的定时变更"},
        503: {"description": "服务端未配置 ADMIN_KEY_ID / ADMIN_HMAC_SECRET"},
    },
)
def post_policy_rollback(
    body: PolicyRollbackIn,
    conn: sqlite3.Connection = Depends(get_conn),
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    """在同一写事务内把策略恢复到指定历史版本。

    已有桶先按当前策略补充到事务时刻，再按目标版本容量截断；
    恢复成功会追加一条 source=rollback 的新版本（版本号继续递增）。
    """
    with write_tx(conn):
        reserve_nonce(conn, admin.key_id, admin.nonce)
        return quota.rollback_policy(
            conn,
            scope=body.scope,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            version=body.version,
        )


# ------------------------------------------------------------ 策略影响预演


@app.post(
    "/policies/impact-preview",
    response_model=ImpactPreviewOut,
    tags=["策略"],
    responses={
        404: {"description": "目标策略不存在"},
        409: {"description": "目标策略或任一请求命中的策略存在待生效（pending）的定时变更"},
        422: {"description": "at_seconds 越界或未按非递减顺序、cost 非正、route_id 缺少 tenant_id 等"},
    },
)
def post_impact_preview(
    body: ImpactPreviewIn, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """预演「把指定策略替换为候选容量/速率」对未来 0～3600 秒请求序列的影响。

    以同一只读快照把各涉及桶按现行策略补充到预演起点，再构建现行/候选两套
    独立内存状态（候选侧仅替换指定策略，并把该桶起点余额截断到候选容量），
    两侧按各项 at_seconds 逐条补充、判足、扣减，前项余额影响后项，拒绝项不
    扣减。返回逐条两侧的准入结果、命中桶余额与差异。全程只读：不改动策略、
    桶、裁决与事件数据；免认证。
    """
    with read_tx(conn):
        return quota.impact_preview(
            conn,
            scope=body.scope,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            capacity=body.capacity,
            refill_rate=body.refill_rate,
            items=[item.model_dump() for item in body.requests],
        )


# ------------------------------------------------------------ 策略定时切换


@app.post(
    "/policy-changes",
    response_model=PolicyChangeOut,
    tags=["策略定时切换"],
    responses={
        401: {"description": "缺少/无效的 HMAC 签名，或时间戳偏离超过 300 秒"},
        404: {"description": "目标策略不存在"},
        409: {"description": "nonce 重用；或 change_id 已存在；或同一策略已有待生效变更"},
        503: {"description": "服务端未配置 ADMIN_KEY_ID / ADMIN_HMAC_SECRET"},
    },
)
def post_policy_change(
    body: PolicyChangeIn,
    conn: sqlite3.Connection = Depends(get_conn),
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    """为已存在的策略登记一条未来生效的定时切换。

    同一策略至多一条 pending 变更；存在 pending 期间 PUT /policies 更新该
    策略会返回 409。到期后无需后台任务：下一次命中该策略的裁决会在同一
    BEGIN IMMEDIATE 事务内先应用变更，再执行裁决。
    """
    with write_tx(conn):
        reserve_nonce(conn, admin.key_id, admin.nonce)
        return quota.create_policy_change(
            conn,
            change_id=body.change_id,
            scope=body.scope,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            capacity=body.capacity,
            refill_rate=body.refill_rate,
            low_balance_threshold=body.low_balance_threshold,
            effective_at=body.effective_at,
        )


@app.get(
    "/policy-changes",
    response_model=list[PolicyChangeOut],
    tags=["策略定时切换"],
)
def get_policy_changes(
    status: str | None = Query(default=None, pattern="^(pending|applied|cancelled)$"),
    scope: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
    route_id: str | None = Query(default=None),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    """列出策略定时变更，可按 status / 策略键过滤。只读。"""
    return quota.list_policy_changes(
        conn,
        status=status,
        scope=scope,
        tenant_id=tenant_id,
        route_id=route_id,
    )


@app.get(
    "/policy-changes/{change_id}",
    response_model=PolicyChangeOut,
    tags=["策略定时切换"],
    responses={404: {"description": "变更不存在"}},
)
def get_policy_change(
    change_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    return quota.get_policy_change(conn, change_id)


@app.delete(
    "/policy-changes/{change_id}",
    response_model=PolicyChangeOut,
    tags=["策略定时切换"],
    responses={
        401: {"description": "缺少/无效的 HMAC 签名，或时间戳偏离超过 300 秒"},
        404: {"description": "变更不存在"},
        409: {"description": "nonce 重用；或变更已生效或已到生效时刻，无法取消"},
        503: {"description": "服务端未配置 ADMIN_KEY_ID / ADMIN_HMAC_SECRET"},
    },
)
def delete_policy_change(
    change_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    """在生效前取消一条 pending 变更；重复取消幂等。"""
    with write_tx(conn):
        reserve_nonce(conn, admin.key_id, admin.nonce)
        return quota.cancel_policy_change(conn, change_id)


# ---------------------------------------------------------------- 桶观察


@app.get("/buckets", response_model=list[BucketOut], tags=["桶状态"])
def get_buckets(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    """查看所有已创建桶的当前（补充后）余额，便于排障。只读，不落库。"""
    return quota.list_buckets(conn)


# ---------------------------------------------------------------- 事件


@app.get("/events", response_model=list[LowBalanceEventOut], tags=["事件"])
def get_events(
    after_id: int = Query(default=0, ge=0, description="游标：只返回 id 大于它的事件"),
    limit: int = Query(default=100, ge=1, le=1000, description="每页最多返回条数"),
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[dict]:
    """按 id 稳定递增分页返回 low_balance 事件。

    用法：先取 `GET /events?limit=N`，然后以最后一条的 id 作为
    `after_id` 继续翻页，直到返回空列表。
    """
    return quota.list_events(conn, after_id=after_id, limit=limit)


# ---------------------------------------------------------------- 裁决 / 撤销


@app.post(
    "/decide",
    response_model=DecisionOut,
    tags=["裁决"],
    responses={409: {"description": "request_id 已存在但载荷冲突"}},
)
def decide(body: DecideIn, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    with write_tx(conn):
        result, _replayed = quota.decide(
            conn,
            request_id=body.request_id,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            cost=body.cost,
        )
        return result


@app.post(
    "/decide-batches",
    response_model=DecideBatchOut,
    tags=["裁决"],
    responses={
        404: {"description": "批次中某条目没有命中任何已配置的配额策略"},
        409: {"description": "batch_id 已存在但完整载荷不一致"},
    },
)
def decide_batches(
    body: DecideBatchIn, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """在一个 BEGIN IMMEDIATE 事务内对 1～50 个请求做整批裁决。

    事务内先以同一时刻应用所有条目命中且到期的策略切换，再按数组顺序
    演算共享的多级桶（前项扣减后的余额供后项使用）。仅全部获准时才写入
    桶余额、普通裁决（decisions）与低余额事件；任一项不足或 request_id
    已存在时这些业务数据一律不变，返回 committed=false、failed_index 和
    该项当时的桶状态。无论成败批次结果都按 batch_id 存档；同键异载荷 409。
    成功批次中的单笔裁决可经 POST /decide 重放、经 /revoke 撤销。
    """
    items = [item.model_dump() for item in body.requests]
    try:
        with write_tx(conn):
            return quota.decide_batch(conn, batch_id=body.batch_id, items=items)
    except quota.BatchRejected as rej:
        # 业务事务已随异常整体回滚（桶/裁决/事件均未改变）；批次档案在
        # 独立事务中保存，保证「无论成败均保存批次结果」
        with write_tx(conn):
            return quota.record_failed_batch(
                conn,
                batch_id=body.batch_id,
                items=items,
                failure=rej.failure,
            )


@app.post(
    "/decide/{request_id}/revoke",
    response_model=RevokeOut,
    tags=["裁决"],
    responses={
        404: {"description": "裁决不存在"},
        409: {"description": "裁决本身被拒绝，无需撤销"},
    },
)
def revoke_decision(
    request_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    with write_tx(conn):
        result, _duplicate = quota.revoke(conn, request_id)
        return result


# ---------------------------------------------------------------- 配额预占


@app.post(
    "/reservations",
    response_model=ReservationOut,
    tags=["配额预占"],
    responses={
        404: {"description": "没有命中任何已配置的配额策略"},
        409: {"description": "reservation_id 已存在但载荷冲突"},
        422: {"description": "cost 非正、ttl_seconds 不在 1～300、只给 route_id 等"},
    },
)
def create_reservation(body: ReservationIn, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    """创建配额预占（免认证）。

    在同一 BEGIN IMMEDIATE 事务、同一时刻内：先应用命中且到期的策略切换、
    把全部到期 pending 预占转 expired 并返还，再补充并判定全部命中桶——
    都充足才统一预扣 cost 并存档原命中桶（pending）；任一不足则一个都不扣，
    存档为 rejected。相同 reservation_id + 相同载荷（tenant/route/cost/ttl）
    重放原结果，异载荷返回 409。
    """
    with write_tx(conn):
        return quota.create_reservation(
            conn,
            reservation_id=body.reservation_id,
            tenant_id=body.tenant_id,
            route_id=body.route_id,
            cost=body.cost,
            ttl_seconds=body.ttl_seconds,
        )


@app.post(
    "/reservations/{reservation_id}/confirm",
    response_model=ReservationOut,
    tags=["配额预占"],
    responses={
        404: {"description": "预占不存在"},
        409: {"description": "预占已到期（expired）/ 已取消 / 已拒绝，无法确认"},
    },
)
def confirm_reservation(
    reservation_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """确认预占：仅把 pending 转为 confirmed，不再扣减任何余额。

    确认与到期竞争时以事务内时间为准，到期优先（先转 expired 并返还，再 409）。
    重复确认幂等，不改变余额；其他终态上的确认返回 409。
    """
    with write_tx(conn):
        return quota.confirm_reservation(conn, reservation_id)


@app.post(
    "/reservations/{reservation_id}/cancel",
    response_model=ReservationOut,
    tags=["配额预占"],
    responses={
        404: {"description": "预占不存在"},
        409: {"description": "预占已确认 / 已到期 / 已拒绝，无法取消"},
    },
)
def cancel_reservation(
    reservation_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """取消预占：pending 转为 cancelled，并在同一事务内向原命中桶返还 cost。

    返还先按当前策略补充/应用到期定时切换再加回，最终不超过桶当前容量。
    重复取消幂等（不再次返还）；已到期的预占先按到期优先转为 expired 再 409；
    confirmed / rejected 上的取消返回 409。
    """
    with write_tx(conn):
        return quota.cancel_reservation(conn, reservation_id)


@app.post(
    "/reservations/{reservation_id}/settle",
    response_model=ReservationOut,
    tags=["配额预占"],
    responses={
        404: {"description": "预占不存在"},
        409: {
            "description": (
                "预占已到期（expired，到期优先：已先持久化 expired 与返还）/"
                "已确认 / 已取消 / 已拒绝；或已按不同 actual_cost 结算；"
                "或实际用量多于预占量但某原桶补充后余额不足补扣差额"
                "（保持 pending、不扣款）"
            )
        },
        422: {"description": "actual_cost 为负数、NaN 或 Infinity"},
    },
)
def settle_reservation(
    reservation_id: str,
    body: ReservationSettleIn,
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    """结算预占：仅未到期的 pending 可结算为 settled，按实际用量「多退少补」。

    分两个串行写事务（同一连接）：
      1. 先在独立事务内把全部到期 pending 预占（可能含本预占）转 expired 并返还，
         单独提交——到期优先：本预占到期时随后返回 409，但 expired 与返还已持久化；
      2. 结算事务沿用创建时保存的原命中桶键与现有补充、定时切换规则：
         actual_cost < cost 向各原桶返还差额（按当前容量封顶）；
         actual_cost > cost 时所有原桶补充后都足够才统一补扣差额，任一不足返回
         409，事务整体回滚，预占保持 pending 且不扣款；actual_cost == cost 不净扣补。
    已 settled：相同 actual_cost 重放原结果（replayed=true），异值 409；
    其他终态 409；未知 ID 404。结算与确认、取消并发时由 BEGIN IMMEDIATE 串行化，
    条件更新保证只有一个终态转换成功。免认证。
    """
    with write_tx(conn):
        ts = quota.expire_due_reservations(conn)
    with write_tx(conn):
        return quota.settle_reservation(
            conn, reservation_id, body.actual_cost, now=ts
        )


@app.get(
    "/reservations/{reservation_id}",
    response_model=ReservationOut,
    tags=["配额预占"],
    responses={404: {"description": "预占不存在"}},
)
def get_reservation(
    reservation_id: str, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """查询预占状态与到期时间；若该预占恰好到期，在同一写事务内先清理为
    expired 并向原桶返还，再返回终态。终态行可反复查询，不改变余额。
    """
    with write_tx(conn):
        return quota.get_reservation(conn, reservation_id)


# ------------------------------------------------------------ 路由桶令牌移拨


@app.post(
    "/route-transfers",
    response_model=RouteTransferOut,
    tags=["路由令牌移拨"],
    responses={
        401: {"description": "缺少/无效的 HMAC 签名，或时间戳偏离超过 300 秒"},
        404: {"description": "源或目标路由策略未配置"},
        409: {
            "description": (
                "transfer_id 已存在但载荷不同；或源桶余额不足；或目标桶余额加移拨量"
                "超过容量（不移拨扣增，但已到期的预占返还与策略切换仍生效）"
            )
        },
        422: {"description": "amount 非正数/NaN/Infinity、源目标路由相同、缺少 tenant_id 等"},
        503: {"description": "服务端未配置 ADMIN_KEY_ID / ADMIN_HMAC_SECRET"},
    },
)
def post_route_transfer(
    body: RouteTransferIn,
    conn: sqlite3.Connection = Depends(get_conn),
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    """在同一租户的两个**不同**路由配额桶之间移拨现有令牌（管理写接口，需签名）。

    分两个串行写事务（同一连接，与预占结算同一「到期优先」约定）：
      1. 独立前序事务：先按现有规则应用两侧已到期的定时策略切换、把全部到期
         pending 预占转 expired 并返还，再把源/目标两桶补充到同一时刻并提交
         ——即使随后移拨返回 409，这些到期效果也已持久化；
      2. 移拨事务（nonce 与业务写入同事务提交）：两侧路由策略必须均已配置
         （否则 404；未创建的桶按各自容量满额处理），源桶补充后余额不足、
         或目标桶补充后余额加移拨量超过容量时返回 409 且一个令牌都不扣增；
         成功时只扣源路由桶、增目标路由桶，不扣全局/租户桶，不写裁决，
         也不产生 low_balance 事件。

    相同 transfer_id + 相同载荷（tenant/source/target/amount）重放原成功结果
    （replayed=true，不再次移拨）；相同编号异载荷返回 409。
    """
    with write_tx(conn):
        quota.prepare_route_transfer(
            conn,
            tenant_id=body.tenant_id,
            source_route_id=body.source_route_id,
            target_route_id=body.target_route_id,
        )
    with write_tx(conn):
        reserve_nonce(conn, admin.key_id, admin.nonce)
        return quota.route_transfer(
            conn,
            transfer_id=body.transfer_id,
            tenant_id=body.tenant_id,
            source_route_id=body.source_route_id,
            target_route_id=body.target_route_id,
            amount=body.amount,
            now=quota.now_utc(),
        )


# ---------------------------------------------------------------- 配额规划


@app.post(
    "/quota-plans",
    response_model=QuotaPlanOut,
    tags=["配额规划"],
    responses={
        422: {"description": "项数非 1～18、item_id 重复/为空、cost 非正、value 为负或非整数、route_id 缺少 tenant_id 等"},
    },
)
def post_quota_plan(
    body: QuotaPlanIn, conn: sqlite3.Connection = Depends(get_conn)
) -> dict:
    """为 1～18 个待执行请求规划配额内总收益最大的子集（免认证、全程只读）。

    在同一只读快照、同一时刻下：按现行策略把各命中桶补充到快照时刻
    （未建桶按满额），不应用到期的定时切换、不推进预占到期状态；每个入选项
    在其所有命中桶（全局 -> 租户 -> 路由，未配置的层不限制）占用 cost，
    同一桶的合计占用不得超过快照余额。在此约束下求精确最优子集：总收益最大；
    并列时项数更多者优先；再并列时升序 ID 列表字典序更小者优先。
    返回入选/未入选 ID、总收益与各命中桶扣除入选项后的剩余额；无可行项时
    返回空集。规划结果不改动任何落库状态，也不保证之后的真实裁决准入。
    """
    with read_tx(conn):
        return quota.plan_quota(
            conn, items=[item.model_dump() for item in body.items]
        )


# ------------------------------------------------------------ 配额消耗核对（只读）


@app.get(
    "/quota-usage",
    response_model=QuotaUsageOut,
    tags=["配额消耗核对"],
    responses={
        422: {"description": "start/end 时间格式无效（非 ISO 8601 或不带时区），或起点不早于终点"},
    },
)
def get_quota_usage(
    start: str = Query(
        ...,
        description="统计窗口起点（UTC，左闭），ISO 8601 显式带时区，如 2026-09-27T12:00:00Z",
    ),
    end: str = Query(
        ...,
        description="统计窗口终点（UTC，右开），ISO 8601 显式带时区，必须晚于 start",
    ),
    conn: sqlite3.Connection = Depends(get_conn),
) -> dict:
    """按租户与路由核对配额请求的实际消耗（免认证、全程只读）。

    输入 UTC 左闭右开窗口 [start, end)：裁决按 decided_at、预占按 created_at
    筛选。在**同一只读快照**内按 tenant_id、route_id 汇总：

    - 已消耗量 `consumed`：获准且未撤销的单笔及成功批次裁决 cost；
      `confirmed` 预占按预占量 cost、`settled` 预占按 actual_cost；
    - 未完成占用量 `held`：报表快照时刻仍未到期的 `pending` 预占 cost
      （到期但尚未惰性清扫的 pending 也按到期处理，不计入）；
    - 被拒绝或已撤销的裁决、`rejected` / `cancelled` / `expired` 预占均不计入；
      重放不产生新行、批次内各笔以自身 request_id 存档，故不重复计量。

    查询只执行 SELECT：不推进预占到期状态，也不修改任何预占或桶。
    无匹配记录时 `groups` 为空列表。时间格式无效或 start >= end 返回 422。
    """
    start_dt = _parse_utc_window_param("start", start)
    end_dt = _parse_utc_window_param("end", end)
    if not start_dt < end_dt:
        raise HTTPException(
            422, "时间窗口非法：start 必须严格早于 end（左闭右开 [start, end)）"
        )
    with read_tx(conn):
        return quota.quota_usage(conn, start=start_dt, end=end_dt)
