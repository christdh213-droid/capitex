"""Deterministic beginner wording and indicator-aware context cache contracts."""
import json
import pandas as pd
import pytest
import database,market_context as mc
from indicators import technical_takeaway
from test_upgrade import db

def states(rsi=54.1,bullish=True,vol='Normal'):
    tone='bullish' if bullish else 'bearish'
    return dict(values=dict(RSI_14=rsi,Close=110 if bullish else 90,SMA_50=100,
        EMA_12=105 if bullish else 95,EMA_26=100,MACD_Hist=1 if bullish else -1),
        tags=[dict(label='Volatility: '+vol,tone='neutral')])

@pytest.mark.parametrize('technical,fragments',[
    (states(24,False,'Elevated'),['leans downward','falling prices','recent fall looks stretched','larger price swings']),
    (states(),['leans upward','rising prices','does not look unusually stretched','usual range']),
    (states(78,True,'Low Squeeze'),['recent rise looks stretched','pause or reverse','unusually quiet']),
    ({},['trend is unclear','not enough history','cannot yet be assessed']),
])
def test_takeaway_regimes_have_exactly_two_plain_sentences(technical,fragments):
    text=technical_takeaway(technical)
    assert len([part for part in text.split('.') if part.strip()])==2
    assert all(fragment in text for fragment in fragments)
    assert not any(jargon in text for jargon in ('RSI','MACD','SMA','EMA'))
    assert text==technical_takeaway(technical)

def test_context_keeps_text_and_refreshes_when_indicators_change(db):
    now=pd.Timestamp('2026-11-23T12:00Z');providers=[('fixture',lambda *_:[])]
    first=mc.get_context('AAPL',now,providers,technical=states(24,False,'Elevated'))
    cached=mc.get_context('AAPL',now,providers,technical=states(24,False,'Elevated'))
    changed=mc.get_context('AAPL',now,providers,technical=states())
    assert first['context_summary_id']==cached['context_summary_id']
    assert first['context_summary_id']!=changed['context_summary_id']
    assert first['technical_summary_text']!=changed['technical_summary_text']
    with database.get_connection() as conn:
        assert conn.execute('SELECT technical_summary_text FROM context_summary_cache WHERE id=?',
            (first['context_summary_id'],)).fetchone()[0]==first['technical_summary_text']
