"""请求 / 响应模型。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PolicyIn(BaseModel):
    scope: Literal["global", "tenant", "route"]
    tenant_id: Optional[str] = Field(default=None, description="scope=tenant/route 时必填")
    route_id: Optional[str] = Field(default=None, description="scope=route 时必填")
    capacity: float = Field(..., ge=0, description="桶容量（令牌数上限）")
    refill_rate: float = Field(..., ge=0, description="每秒补充的令牌数（令牌/秒）")
    low_balance_threshold: Optional[float] = Field(
        default=None,
        description="低余额阈值：null 关闭；否则须在 0 到 capacity 之间。"
        "获准裁决令桶从阈值之上降至不高于阈值时产生 low_balance 事件",
    )


class PolicyOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    capacity: float
    refill_rate: float
    low_balance_threshold: Optional[float]
    updated_at: str


class PolicyRollbackIn(BaseModel):
    scope: Literal["global", "tenant", "route"]
    tenant_id: Optional[str] = Field(default=None, description="scope=tenant/route 时必填")
    route_id: Optional[str] = Field(default=None, description="scope=route 时必填")
    version: int = Field(..., ge=1, description="要恢复到的历史版本号")


class PolicyVersionOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    version: int
    capacity: float
    refill_rate: float
    low_balance_threshold: Optional[float]
    source: Literal["put", "scheduled", "rollback", "baseline"]
    change_id: Optional[str] = None
    restored_from_version: Optional[int] = None
    created_at: str


class PolicyChangeIn(BaseModel):
    change_id: str = Field(..., min_length=1, description="变更幂等键，全局唯一")
    scope: Literal["global", "tenant", "route"]
    tenant_id: Optional[str] = Field(default=None, description="scope=tenant/route 时必填")
    route_id: Optional[str] = Field(default=None, description="scope=route 时必填")
    capacity: float = Field(..., ge=0, description="生效后的新桶容量")
    refill_rate: float = Field(..., ge=0, description="生效后的新补充速率（令牌/秒）")
    low_balance_threshold: Optional[float] = Field(
        default=None,
        description="生效后的新低余额阈值：null 关闭；否则须在 0 到 capacity 之间",
    )
    effective_at: datetime = Field(
        ...,
        description="生效时刻（UTC），必须晚于当前时间；支持带偏移的 ISO 8601 时间",
    )

    @field_validator("effective_at")
    @classmethod
    def _effective_must_be_utc(cls, v: datetime) -> datetime:
        # 不带时区的时间无法可靠解释为 UTC，拒绝并提示显式带 Z / +00:00
        if v.tzinfo is None:
            raise ValueError("effective_at 必须显式带时区（UTC，例如 2026-10-01T00:00:00Z）")
        return v.astimezone(timezone.utc)


class PolicyChangeOut(BaseModel):
    change_id: str
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    capacity: float
    refill_rate: float
    low_balance_threshold: Optional[float]
    effective_at: str
    status: Literal["pending", "applied", "cancelled"]
    created_at: str
    applied_at: Optional[str] = None
    cancelled_at: Optional[str] = None


class DecideIn(BaseModel):
    request_id: str = Field(..., min_length=1, description="幂等键，相同 request_id 重放原结果")
    tenant_id: Optional[str] = None
    route_id: Optional[str] = Field(default=None, description="提供 route_id 时必须带 tenant_id")
    cost: float = Field(..., gt=0, description="本次请求需要的令牌数，所有命中桶统一扣减")


class BucketOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    capacity: float
    refill_rate: float
    balance: float


class DecisionOut(BaseModel):
    request_id: str
    allowed: bool
    cost: float
    tenant_id: Optional[str]
    route_id: Optional[str]
    decided_at: str
    retry_at: Optional[str]
    retry_after_seconds: Optional[float]
    retry_possible: bool
    replayed: bool = False
    revoked: bool = False
    revoked_at: Optional[str] = None
    buckets: list[BucketOut]


class DecideBatchItemIn(BaseModel):
    request_id: str = Field(..., min_length=1, description="单笔裁决幂等键，写入 decisions 表，可由 /decide 重放、/revoke 撤销")
    tenant_id: Optional[str] = None
    route_id: Optional[str] = Field(default=None, description="提供 route_id 时必须带 tenant_id")
    cost: float = Field(..., gt=0, description="本次请求需要的令牌数，所有命中桶统一扣减")


class DecideBatchIn(BaseModel):
    batch_id: str = Field(..., min_length=1, description="批次幂等键：相同 batch_id + 相同完整载荷重放原结果；同键异载荷返回 409")
    requests: list[DecideBatchItemIn] = Field(
        ..., min_length=1, max_length=50, description="1～50 个裁决请求，按数组顺序演算共享桶"
    )


class BatchDecisionOut(BaseModel):
    """成功批次中的单笔裁决（结构与 POST /decide 的结果一致，replayed 恒为 false）。"""
    index: int
    request_id: str
    allowed: bool
    cost: float
    tenant_id: Optional[str]
    route_id: Optional[str]
    decided_at: str
    buckets: list[BucketOut]


class BatchFailureOut(BaseModel):
    index: int = Field(..., description="数组中首个令整批失败的项下标（从 0 开始）")
    request_id: str
    reason: Literal["insufficient_balance", "request_id_exists"]
    cost: float
    tenant_id: Optional[str]
    route_id: Optional[str]
    buckets: list[BucketOut] = Field(
        ..., description="失败那一刻该条目命中桶的状态（前序条目已模拟扣减，余额未持久化）"
    )


class DecideBatchOut(BaseModel):
    batch_id: str
    committed: bool
    replayed: bool = False
    decided_at: str
    results: Optional[list[BatchDecisionOut]] = Field(
        default=None, description="committed=true：按请求顺序的各单笔裁决"
    )
    failed_index: Optional[int] = None
    failure: Optional[BatchFailureOut] = Field(
        default=None, description="committed=false：首个失败条目及其当时桶状态"
    )


class RevokeOut(BaseModel):
    request_id: str
    allowed: bool
    revoked: bool
    revoked_at: Optional[str]
    replayed: bool
    cost: float
    buckets: list[BucketOut]


class ImpactPreviewItemIn(BaseModel):
    tenant_id: Optional[str] = None
    route_id: Optional[str] = Field(default=None, description="提供 route_id 时必须带 tenant_id")
    cost: float = Field(..., gt=0, description="本次请求需要的令牌数，所有命中桶统一扣减")
    at_seconds: float = Field(
        ..., ge=0, le=3600,
        description="相对预演起点的秒数（0～3600）；多条请求必须按非递减顺序提交",
    )


class ImpactPreviewIn(BaseModel):
    scope: Literal["global", "tenant", "route"]
    tenant_id: Optional[str] = Field(default=None, description="scope=tenant/route 时必填")
    route_id: Optional[str] = Field(default=None, description="scope=route 时必填")
    capacity: float = Field(..., ge=0, description="候选桶容量（仅替换指定策略）")
    refill_rate: float = Field(..., ge=0, description="候选补充速率（令牌/秒）")
    requests: list[ImpactPreviewItemIn] = Field(
        ..., min_length=1, max_length=50,
        description="1～50 条预演请求，按 at_seconds 非递减顺序",
    )


class ImpactPreviewSideOut(BaseModel):
    """预演单侧（现行/候选）对一条请求的裁决结果与命中桶余额。"""
    allowed: bool
    buckets: list[BucketOut]


class ImpactPreviewBucketDiffOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    current_balance: float
    candidate_balance: float
    balance_delta: float = Field(..., description="候选侧余额 - 现行侧余额")


class ImpactPreviewItemOut(BaseModel):
    index: int
    at_seconds: float
    tenant_id: Optional[str]
    route_id: Optional[str]
    cost: float
    current: ImpactPreviewSideOut
    candidate: ImpactPreviewSideOut
    allowed_changed: bool = Field(..., description="两侧准入结果是否不同")
    bucket_diffs: list[ImpactPreviewBucketDiffOut] = Field(
        ..., description="逐项命中桶的两侧余额对比（与 buckets 同序）"
    )


class ImpactPreviewSummaryOut(BaseModel):
    total_items: int
    current_allowed: int = Field(..., description="现行策略下获准的条数")
    candidate_allowed: int = Field(..., description="候选策略下获准的条数")
    allowed_changed_items: int = Field(..., description="两侧准入结果不同的条数")


class ImpactPreviewOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    current_capacity: float
    current_refill_rate: float
    candidate_capacity: float
    candidate_refill_rate: float
    started_at: str = Field(..., description="预演起点（at_seconds=0 对应的 UTC 时刻）")
    items: list[ImpactPreviewItemOut]
    summary: ImpactPreviewSummaryOut


class ReservationIn(BaseModel):
    reservation_id: str = Field(
        ..., min_length=1, description="预占幂等键：相同 ID + 相同载荷重放原结果；异载荷 409"
    )
    tenant_id: Optional[str] = None
    route_id: Optional[str] = Field(default=None, description="提供 route_id 时必须带 tenant_id")
    cost: float = Field(..., gt=0, description="预占令牌数（正数），全部命中桶统一预扣")
    ttl_seconds: int = Field(
        ..., ge=1, le=300, description="预占有效期（1～300 秒）；到期未确认自动失效并返还"
    )


class ReservationSettleIn(BaseModel):
    actual_cost: float = Field(
        ..., description="实际用量（令牌数）：必须为有限非负数（含 0）"
    )

    @field_validator("actual_cost", mode="before")
    @classmethod
    def _actual_cost_finite_non_negative(cls, v):
        # ge=0 无法拒绝 NaN（NaN 比较恒为假）；非有限浮点在此显式拒绝为 422。
        # 422 响应回显的非有限 input 由 main.py 的校验异常处理器统一安全序列化。
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("actual_cost 必须是非负数字")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("actual_cost 必须是有限数（不接受 NaN / Infinity）")
        if f < 0:
            raise ValueError("actual_cost 必须非负")
        return f


class ReservationOut(BaseModel):
    reservation_id: str
    status: Literal[
        "pending", "confirmed", "cancelled", "expired", "rejected", "settled"
    ]
    allowed: bool = Field(
        ..., description="预占是否成功（status != rejected）；rejected 表示创建时桶不足，未扣余额"
    )
    tenant_id: Optional[str]
    route_id: Optional[str]
    cost: float
    ttl_seconds: int
    created_at: str
    expire_at: str = Field(..., description="到期时刻（ISO 8601 UTC）；到期未确认自动 expired 并返还")
    confirmed_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    expired_at: Optional[str] = None
    actual_cost: Optional[float] = Field(
        default=None, description="结算实际用量；仅 status=settled 时非空"
    )
    settled_at: Optional[str] = Field(
        default=None, description="结算完成时刻（ISO 8601 UTC）；仅 settled 时非空"
    )
    replayed: bool = False
    buckets: list[BucketOut] = Field(
        ..., description="原命中桶按当前策略补充后的实时余额（只读快照）"
    )


class RouteTransferIn(BaseModel):
    transfer_id: str = Field(
        ..., min_length=1,
        description="移拨唯一编号：相同编号 + 相同载荷重放原成功结果；编号相同载荷不同返回 409",
    )
    tenant_id: str = Field(..., min_length=1, description="租户 ID；源/目标路由必须同属该租户")
    source_route_id: str = Field(..., min_length=1, description="源路由 ID（扣出令牌的路由桶）")
    target_route_id: str = Field(..., min_length=1, description="目标路由 ID（接收令牌的路由桶）")
    amount: float = Field(..., description="移拨令牌数：必须为有限正数")

    @field_validator("amount", mode="before")
    @classmethod
    def _amount_finite_positive(cls, v):
        # gt=0 无法拒绝 NaN（NaN 比较恒为假）；非有限浮点在此显式拒绝为 422。
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("amount 必须是数字")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("amount 必须是有限数（不接受 NaN / Infinity）")
        if f <= 0:
            raise ValueError("amount 必须为正数")
        return f

    @field_validator("target_route_id")
    @classmethod
    def _routes_must_differ(cls, v, info):
        src = info.data.get("source_route_id")
        if src is not None and v == src:
            raise ValueError("source_route_id 与 target_route_id 必须是不同的路由")
        return v


class RouteTransferBucketOut(BaseModel):
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    capacity: float
    refill_rate: float
    balance: float = Field(..., description="移拨后的桶余额（已按当前策略补充并扣/增移拨量）")


class RouteTransferOut(BaseModel):
    transfer_id: str
    tenant_id: str
    source_route_id: str
    target_route_id: str
    amount: float
    transferred_at: str = Field(..., description="移拨时刻（ISO 8601 UTC）")
    replayed: bool = False
    source: RouteTransferBucketOut = Field(..., description="源路由桶移拨后余额")
    target: RouteTransferBucketOut = Field(..., description="目标路由桶移拨后余额")


class QuotaPlanItemIn(BaseModel):
    item_id: str = Field(..., min_length=1, description="待执行请求的标识，本次规划内唯一")
    tenant_id: str = Field(..., min_length=1, description="必填；命中租户层策略")
    route_id: Optional[str] = Field(
        default=None, description="可选；提供时同时命中路由层策略（必须带 tenant_id）"
    )
    cost: float = Field(..., gt=0, description="入选后在所有命中桶占用的令牌数（正数）")
    value: int = Field(..., ge=0, description="入选收益（非负整数）")


class QuotaPlanIn(BaseModel):
    items: list[QuotaPlanItemIn] = Field(
        ..., min_length=1, max_length=18,
        description="1～18 个待执行请求；在只读快照上选出配额内总收益最大的子集",
    )


class QuotaPlanOut(BaseModel):
    selected_item_ids: list[str] = Field(..., description="入选项 ID（升序）")
    unselected_item_ids: list[str] = Field(..., description="未入选项 ID（升序）")
    total_value: int = Field(..., description="入选项的总收益")
    planned_at: str = Field(..., description="规划快照时刻（ISO 8601 UTC）")
    buckets: list[BucketOut] = Field(
        ..., description="本次规划命中的桶扣除入选项占用后的剩余额（只读演算，不落库）"
    )


class UsageGroupOut(BaseModel):
    """按 tenant_id + route_id 汇总的已消耗量与未完成占用量。

    分组键取自裁决 / 预占请求自身携带的 tenant_id / route_id：
    仅带 tenant_id 的请求 route_id 为 null，两者都不带（全局级请求）均为 null。
    """

    tenant_id: Optional[str] = Field(..., description="租户 ID；全局级分组为 null")
    route_id: Optional[str] = Field(..., description="路由 ID；租户级 / 全局级分组为 null")
    consumed: float = Field(
        ...,
        description=(
            "已消耗量：窗口内获准且未撤销的单笔及批次裁决 cost 合计，"
            "加 status=confirmed 预占的 cost、status=settled 预占的 actual_cost"
        ),
    )
    held: float = Field(
        ...,
        description=(
            "未完成占用量：status=pending 且在报表时刻（同一只读快照）尚未到期的"
            "预占 cost 合计；已到期（含到期但尚未惰性清扫）的 pending 不计入"
        ),
    )
    decision_count: int = Field(..., description="计入 consumed 的获准裁决笔数（不含被拒绝/已撤销）")
    confirmed_count: int = Field(..., description="计入 consumed 的 confirmed 预占笔数")
    settled_count: int = Field(..., description="计入 consumed 的 settled 预占笔数")
    pending_count: int = Field(..., description="计入 held 的未到期 pending 预占笔数")


class QuotaUsageOut(BaseModel):
    start: str = Field(..., description="统计窗口起点（UTC，左闭）")
    end: str = Field(..., description="统计窗口终点（UTC，右开）")
    evaluated_at: str = Field(..., description="同一只读快照的报表时刻（ISO 8601 UTC）")
    total_consumed: float = Field(..., description="全部分组已消耗量合计")
    total_held: float = Field(..., description="全部分组未完成占用量合计")
    groups: list[UsageGroupOut] = Field(
        ..., description="按 tenant_id、route_id 汇总的分组（按范围全局→租户→路由排序）"
    )


class LowBalanceEventOut(BaseModel):
    id: int = Field(..., description="单调递增的事件 id，用作分页游标")
    scope: str
    tenant_id: Optional[str]
    route_id: Optional[str]
    balance_before: float = Field(..., description="扣减前（时间补充后）的余额")
    balance_after: float = Field(..., description="扣减后的余额")
    threshold: float = Field(..., description="触发本事件时的低余额阈值")
    request_id: str = Field(..., description="触发本事件的裁决 request_id")
    created_at: str
