"""Consistent CAPITEX display formatting; stored timestamps remain UTC."""
from datetime import datetime, timezone
import math
from zoneinfo import ZoneInfo


def signed_money(value):
    if value is None or not math.isfinite(float(value)):
        return '—'
    value = float(value)
    return f'{"-" if value < 0 else "+"}${abs(value):,.2f}'


def local_timestamp(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')
    except (ValueError, TypeError):
        return '—'


def performance_text(summary):
    closed = int(summary.get('total_closed_trades', 0))
    ratio = summary.get('win_loss_ratio')
    value = f'{ratio:.2f} W/L' if closed and ratio is not None else '—'
    win = summary.get('win_rate_pct')
    details = f'Win: {win:.1f}%' if closed and win is not None else 'Win: —'
    drawdown = summary.get('max_drawdown_pct')
    if summary.get('snapshots') and drawdown is not None:
        details += f' | MDD: {drawdown:.2f}%'
    return value, details
