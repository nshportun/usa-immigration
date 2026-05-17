"""
Budget monitor for Account 1.

Polls AWS Cost Explorer when available. If the IAM user lacks ce:GetCostAndUsage
permission, logs a warning and continues (does not block the pipeline).

To enable Cost Explorer checks, attach the AWS managed policy
'AWSBillingReadOnlyAccess' or 'CostExplorerReadOnlyAccess' to the IAM user.

Raises BudgetExceeded only when Cost Explorer is accessible and spend >= limit.
"""

import datetime
import sys
import time

import structlog

from scripts.aws_config import BUDGET_LIMIT_USD, ce_client

log = structlog.get_logger()


class BudgetExceeded(Exception):
    pass


def get_current_month_spend() -> float | None:
    """
    Return month-to-date spend in USD.
    Returns None if Cost Explorer is not accessible (permission missing).
    """
    try:
        ce = ce_client()
        today = datetime.date.today()
        start = today.replace(day=1).isoformat()
        end = today.isoformat()
        if start == end:
            return 0.0
        resp = ce.get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
        )
        results = resp.get("ResultsByTime", [])
        if not results:
            return 0.0
        return float(results[0]["Total"]["UnblendedCost"]["Amount"])
    except Exception as e:
        if "AccessDenied" in str(e) or "AccessDeniedException" in str(e):
            log.warning(
                "budget_ce_permission_missing",
                msg="IAM user lacks ce:GetCostAndUsage — budget enforcement disabled. "
                    "Attach AWSBillingReadOnlyAccess to enable it.",
            )
            return None
        raise


def check_budget(label: str = "") -> float | None:
    """
    Check current spend. Raises BudgetExceeded if at or over limit.
    Returns current spend (or None if Cost Explorer unavailable).
    """
    spend = get_current_month_spend()
    if spend is None:
        log.warning("budget_check_skipped", label=label, reason="no_ce_permission")
        return None

    log.info("budget_check", label=label, spend_usd=round(spend, 2), limit_usd=BUDGET_LIMIT_USD)
    if spend >= BUDGET_LIMIT_USD:
        raise BudgetExceeded(
            f"Account 1 spend ${spend:.2f} has reached the ${BUDGET_LIMIT_USD:.2f} limit. "
            "Stop pipeline and switch to Account 2 for Bedrock compute."
        )
    remaining = BUDGET_LIMIT_USD - spend
    if remaining < 10:
        log.warning("budget_low", remaining_usd=round(remaining, 2))
    return spend


def budget_guard(interval_minutes: int = 30):
    """Decorator: checks budget before the decorated function runs."""
    import functools

    def decorator(fn):
        last_check = [0.0]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            now = time.time()
            if now - last_check[0] >= interval_minutes * 60:
                check_budget(label=fn.__name__)
                last_check[0] = now
            return fn(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    spend = check_budget(label="manual")
    if spend is None:
        print("Cost Explorer unavailable — attach AWSBillingReadOnlyAccess to IAM user to enable budget checks.")
    else:
        print(f"Current spend: ${spend:.2f} / ${BUDGET_LIMIT_USD:.2f}")
