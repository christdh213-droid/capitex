"""Pure 1x-collateral short accounting shared by fills, liquidation and previews."""
from decimal import Decimal, ROUND_HALF_UP

CENT=Decimal('0.01')
ZERO=Decimal(0)

def dec(value):
    result=Decimal(str(value))
    if not result.is_finite():raise ValueError('Short values must be finite.')
    return result

def money(value):
    return dec(value).quantize(CENT,rounding=ROUND_HALF_UP)

def calculate(action,quantity,price,position=None):
    """No I/O or mutation. Monetary cash flows round once to cents, half up."""
    qty=dec(quantity);price=dec(price)
    if qty<Decimal('0.00000001') or price<=0:raise ValueError('Short quantity and price must be positive.')
    old=dict(position) if position is not None else {}
    held=dec(old.get('quantity',0));entry=dec(old.get('avg_entry_price',0))
    locked=money(old.get('locked_collateral',0))
    if action=='SHORT_OPEN':
        if held and price>=2*entry:raise ValueError('Position reached liquidation. Refresh before opening another short.')
        required=money(qty*price)
        if required<=0:raise ValueError('Order collateral rounds below one cent.')
        merged=held+qty
        return dict(action=action,quantity=qty,required_collateral=required,cash_delta=-required,
            realized_pnl=ZERO,remaining_quantity=merged,locked_collateral=locked+required,
            avg_entry_price=(held*entry+qty*price)/merged,would_liquidate=False)
    if action not in ('SHORT_COVER','SHORT_LIQUIDATED'):raise ValueError('Invalid short action.')
    if held<=0 or qty>held:raise ValueError('Cover quantity exceeds the open short position.')
    liquidated=action=='SHORT_LIQUIDATED' or price>=2*entry
    if liquidated or held-qty<Decimal('0.00000001'):qty=held
    released=locked if qty==held else money(locked*qty/held)
    raw_pnl=money((entry-price)*qty)
    returned=ZERO if liquidated else max(ZERO,money(released+raw_pnl))
    return dict(action='SHORT_LIQUIDATED' if liquidated else 'SHORT_COVER',quantity=qty,
        required_collateral=ZERO,released_collateral=released,cash_delta=returned,
        realized_pnl=money(returned-released),remaining_quantity=held-qty,
        locked_collateral=money(locked-released),avg_entry_price=entry,would_liquidate=liquidated)
