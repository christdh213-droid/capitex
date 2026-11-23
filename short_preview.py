"""Read-only wallet preview; every short cash flow comes from short_accounting."""
import database,data_loader
from short_accounting import calculate,dec,money

def preview(symbol,quantity,entry_price,pct_move,market_provider=None):
    symbol=data_loader._symbol(symbol)
    entry=dec(entry_price);move=dec(pct_move)
    if entry<=0 or not -30<=move<=120:raise ValueError('Price must be positive and move between -30 and +120 percent.')
    with database.get_connection() as conn:
        cash=dec(conn.execute('SELECT cash FROM wallet WHERE id=1').fetchone()[0])
        longs=[dict(r) for r in conn.execute('SELECT * FROM holdings')]
        shorts=[dict(r) for r in conn.execute('SELECT * FROM short_positions')]
    old=next((r for r in shorts if r['symbol']==symbol),None)
    opened=calculate('SHORT_OPEN',quantity,entry,old)
    if opened['required_collateral']>cash:raise ValueError('Insufficient available cash for 1x short collateral.')
    # Keep quote precision for the 2x trigger; round cash flows, not market prices.
    simulated=entry*(1+move/100)
    position=dict(quantity=opened['remaining_quantity'],avg_entry_price=opened['avg_entry_price'],
                  locked_collateral=opened['locked_collateral'])
    closed=calculate('SHORT_COVER',position['quantity'],simulated,position)
    # Quotes for other assets are fetched after the read connection has closed.
    if market_provider is None:
        from trading_engine import _market_assets
        market_provider=_market_assets
    symbols=list({r['ticker'] for r in longs}|{r['symbol'] for r in shorts if r['symbol']!=symbol})
    markets=market_provider(symbols) if symbols else {}
    estimated=False
    def mark(ticker,fallback):
        nonlocal estimated
        item=markets.get(ticker,{})
        if item.get('available') and item.get('price') is not None:return dec(item['price'])
        estimated=True;return dec(fallback)
    long_value=sum((dec(r['quantity'])*(simulated if r['ticker']==symbol else mark(r['ticker'],r['avg_buy_price'])) for r in longs),dec(0))
    other_short_equity=dec(0)
    for r in shorts:
        if r['symbol']==symbol:continue
        value=calculate('SHORT_COVER',r['quantity'],mark(r['symbol'],r['avg_entry_price']),r)
        other_short_equity+=value['cash_delta']
    pnl=closed['realized_pnl'];locked=opened['locked_collateral']
    equity=money(cash-opened['required_collateral']+long_value+other_short_equity+closed['cash_delta'])
    result=dict(simulated_price=simulated,required_collateral=opened['required_collateral'],
        projected_unrealized_pnl=pnl,projected_roi_pct=pnl/locked*100 if locked else dec(0),
        liquidation_price=2*opened['avg_entry_price'],would_liquidate=closed['would_liquidate'],
        projected_total_equity=equity,valuation_estimated=estimated)
    return {k:float(v) if hasattr(v,'as_tuple') else v for k,v in result.items()}
