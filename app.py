"""CAPITEX consumer wallet and progressive-disclosure AI Predictor."""
from html import escape
from pathlib import Path
import uuid
from time import monotonic
import pandas as pd
import requests
import streamlit as st
import branding
import chart_plotter
import config
import academic
from formatting import signed_money, local_timestamp, performance_text

st.set_page_config(page_title='CAPITEX',page_icon=str(config.LOGO_PATH),layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>'+ (config.ASSETS_DIR/'wallet.css').read_text(encoding='utf-8-sig')+'</style>',unsafe_allow_html=True)
st.markdown('<style>'+ (config.ASSETS_DIR/'predictor.css').read_text(encoding='utf-8-sig')+'</style>',unsafe_allow_html=True)


def api(method,path,payload=None):
    try:
        response=requests.request(method,config.FASTAPI_BASE_URL+path,json=payload,timeout=(3,180 if path.startswith(('/api/predict/','/api/inspect/csv')) else 45))
        if response.status_code>=400:
            detail=response.json().get('detail','Please try again shortly.')
            if isinstance(detail,list): detail='Check the order amount and try again.'
            raise RuntimeError(str(detail))
        return response.json()
    except requests.RequestException as exc:
        raise RuntimeError('We could not reach CAPITEX. Please try again in a moment.') from exc


def caption(text):
    st.caption(str(text).replace('$', r'\$'))


def money(value):
    return '—' if value is None else f'${float(value):,.2f}'


def open_asset(ticker,action=None):
    st.session_state.view='asset'
    st.session_state.asset=ticker
    st.session_state.order_action=action
    st.session_state.order_size=0.
    st.session_state.pop('pending_order',None)


def go_home():
    st.session_state.view='home'
    st.session_state.order_action=None


def choose_assets(action):
    st.session_state.view='choose'
    st.session_state.order_action=action


def asset_card(asset,prefix='market',owned=False,action=None):
    ticker=asset['ticker']
    name=asset['name']
    if owned:
        subtitle=f"{asset['quantity']:,.6f}".rstrip('0').rstrip('.')+f' {ticker}'
        value=money(asset['market_value'])
        change=f"{signed_money(asset['unrealized_pnl'])} / {asset['unrealized_pnl_pct']:+.2f}%"
        positive=asset['unrealized_pnl']>=0
    else:
        subtitle='Crypto' if asset.get('kind')=='crypto' else ticker
        value=money(asset.get('price'))
        pct=asset.get('change_24h_pct')
        change=f'{pct:+.2f}%' if pct is not None else 'Quote unavailable'
        positive=pct is not None and pct>=0
    logo=asset.get('logo') or branding.asset_logo_source(ticker,False)
    with st.container(key=f'assetcard_{prefix}_{ticker}',gap=None):
        st.markdown(f'''<div class="asset-row"><img src="{escape(logo,quote=True)}" alt="{escape(name)} logo"/>
            <div class="asset-name"><b>{escape(name)}</b><small>{escape(subtitle)}</small></div>
            <div class="asset-values"><b>{escape(value)}</b><small class="{'gain' if positive else 'loss' if asset.get('available',True) else 'muted'}">{escape(change)}</small></div>
            <span class="asset-arrow">›</span></div>''',unsafe_allow_html=True)
        st.button(f'Open {name} · {ticker}',key=f'open_{prefix}_{ticker}',width='stretch',
                  on_click=open_asset,args=(ticker,action))


def section(title,note=''):
    st.markdown(f'<div class="section-title"><h3>{escape(title)}</h3><span>{escape(note)}</span></div>',unsafe_allow_html=True)


def render_home(wallet):
    positive=wallet['change_24h']>=0
    period='Past 24h' if wallet['change_period']=='24h' else 'Since start'
    st.markdown(f'''<div class="hero"><div class="eyebrow">Your virtual balance</div>
        <div class="balance">{money(wallet['net_worth'])}</div>
        <span class="change-pill {'gain' if positive else 'loss loss-bg'}">{'↗' if positive else '↘'} {signed_money(wallet['change_24h'])} ({wallet['change_24h_pct']:+.2f}%) <span class="muted">· {period}</span></span>
        <div class="cash-pills"><span>Available cash <b>{money(wallet['cash'])}</b></span>
        <span>Invested <b>{money(wallet['invested_value'])}</b></span></div></div>''',unsafe_allow_html=True)
    if wallet.get('valuation_estimated'):
        st.warning('Some prices are unavailable. Your balance includes cost estimates for those positions.')
    quick=st.columns(4,gap='small')
    with quick[0]: st.button('Buy',icon=':material/add:',key='quick_buy',width='stretch',on_click=choose_assets,args=('BUY',))
    with quick[1]: st.button('Sell',icon=':material/remove:',key='quick_sell',width='stretch',on_click=choose_assets,args=('SELL',))
    with quick[2]:
        if st.button('History',icon=':material/history:',key='quick_history',width='stretch'):
            st.session_state.view='history';st.rerun()
    with quick[3]:
        if st.button('Reset',icon=':material/restart_alt:',key='quick_reset',width='stretch'):
            st.session_state.reset_open=not st.session_state.get('reset_open',False)
    if st.session_state.get('reset_open'):
        with st.container(border=True):
            st.markdown('**Start fresh with $100,000 in virtual cash?**'.replace('$',r'\$'))
            caption('This clears the CAPITEX wallet’s positions and trade history.')
            confirm=st.checkbox('I understand this clears my virtual wallet.',key='reset_confirm')
            if st.button('Reset my virtual wallet',disabled=not confirm,width='stretch'):
                api('POST','/api/wallet/reset')
                st.session_state.reset_open=False
                st.session_state.flash='Your wallet has been reset to $100,000.00.'
                st.rerun()
    section('Your positions',f"{len(wallet['holdings'])} assets")
    if wallet['holdings']:
        for asset in wallet['holdings']: asset_card(asset,'position',True)
    else:
        st.markdown('<div class="empty-state"><div class="empty-symbol">◇</div><strong>Your first position starts here.</strong>Explore the markets below and make a virtual trade.</div>',unsafe_allow_html=True)
    section('Explore markets','Stocks & crypto · USD')
    for asset in wallet['watchlist']: asset_card(asset)
    caption('Quotes may be delayed. Prices and your portfolio refresh every 30 seconds.')


def sync_lookback(source):
    number=int(st.session_state[source])
    st.session_state.lookback_days=number
    st.session_state.lookback_slider=number
    st.session_state.lookback_exact=number
    span={'24H':1,'7D':7,'30D':30,'90D':90}[st.session_state.get('chart_timeframe','90D')]
    if number>span:
        st.session_state.timeframe=next(label for label,days in [('7D',7),('30D',30),('90D',90)] if days>=number)
        st.session_state.chart_timeframe=st.session_state.timeframe


def sync_timeframe():
    st.session_state.chart_timeframe=st.session_state.timeframe
    days={'24H':1,'7D':7,'30D':30,'90D':90}[st.session_state.timeframe]
    if st.session_state.lookback_days>days:
        st.session_state.lookback_days=days
        st.session_state.lookback_exact=days
        st.session_state.lookback_slider=days


def fill_order(fraction,cash,quantity,price):
    action=st.session_state.order_action
    unit=st.session_state.get('order_unit','Dollar Amount')
    if fraction==1:
        unit='Dollar Amount' if action=='BUY' else 'Share / Token Quantity'
        st.session_state.order_unit=unit
    capacity=(cash if action=='BUY' else quantity*price) if unit=='Dollar Amount' else (cash/price if action=='BUY' else quantity)
    st.session_state.order_size=float(quantity if fraction==1 and action=='SELL' else round(capacity*fraction,8))
    st.session_state.pop('pending_order',None)


def change_order_unit():
    st.session_state.order_size=0.
    st.session_state.pop('pending_order',None)


def render_order(asset):
    action=st.session_state.get('order_action')
    if action not in ('BUY','SELL'): return
    holding=asset.get('holding') or {}
    quantity=float(holding.get('quantity',0))
    price=asset['current_price']
    with st.container(border=True,key='order_card'):
        title,close=st.columns([5,1])
        with title: st.markdown(f"#### {'Buy' if action=='BUY' else 'Sell'} {escape(asset['ticker'])}")
        with close:
            if st.button('✕',key='close_order',help='Close order card',width='stretch'):
                st.session_state.order_action=None;st.rerun()
        caption(f"Available cash {money(asset['cash'])} · You own {quantity:,.6f} {asset['ticker']}")
        unit_default={} if 'order_unit' in st.session_state else {'default':'Dollar Amount'}
        unit=st.segmented_control('Order size',options=['Dollar Amount','Share / Token Quantity'],key='order_unit',required=True,width='stretch',on_change=change_order_unit,**unit_default)
        fractions=st.columns(3)
        for col,label,fraction in zip(fractions,['25%','50%','MAX'],[.25,.5,1.]):
            with col: st.button(label,key=f'fill_{label}',width='stretch',on_click=fill_order,
                args=(fraction,asset['cash'],quantity,price))
        size=st.number_input('Dollar amount (USD)' if unit=='Dollar Amount' else 'Share / token quantity',
            min_value=0.,step=10. if unit=='Dollar Amount' else .001,
            format='%.2f' if unit=='Dollar Amount' else '%.8f',key='order_size')
        estimate=size if unit=='Dollar Amount' else size*price
        st.markdown(f'<p class="caption">Estimated {"cost" if action=="BUY" else "proceeds"}: <b>{money(estimate)}</b> · Fill uses a fresh market quote.</p>',unsafe_allow_html=True)
        with st.expander('Position protection',expanded=False):
            protect=st.checkbox('Set trailing stop and take profit for this purchase',value=False,disabled=action=='SELL')
            if protect and action=='BUY':
                sl=st.number_input('Trailing stop (%)',min_value=.1,max_value=100.,value=3.,step=.5)
                tp=st.number_input('Take profit (%)',min_value=.1,max_value=100.,value=6.,step=.5)
            else: sl=tp=None
        if action=='SELL' and quantity<=0: st.info('You do not own this asset yet.')
        if st.button(f"Confirm {action.lower()}",type='primary',width='stretch',disabled=size<=0 or (action=='SELL' and quantity<=0),key='execute_order'):
            fingerprint=(asset['ticker'],action,unit,size,sl,tp)
            pending=st.session_state.get('pending_order')
            if not pending or pending['fingerprint']!=fingerprint:
                pending={'fingerprint':fingerprint,'id':str(uuid.uuid4())}
                st.session_state.pending_order=pending
            payload={'ticker':asset['ticker'],'action':action,'amount' if unit=='Dollar Amount' else 'quantity':size,
                     'client_order_id':pending['id'],'stop_loss_pct':sl,'take_profit_pct':tp}
            result=api('POST','/api/trade',payload)
            fill=result['trade']
            st.session_state.flash=f"{action} complete · {fill['quantity']:,.8f} {fill['ticker']} at {money(fill['price'])} · {fill['execution_label']}"
            st.session_state.pop('pending_order',None)
            st.session_state.order_action=None
            st.rerun()


def render_asset():
    st.button('← Back to Wallet',type='tertiary',on_click=go_home)
    ticker=st.session_state.asset
    asset=api('GET',f'/api/asset/{ticker}?days_ago={st.session_state.lookback_days}')
    quote=asset['quote']
    positive=(quote.get('change_24h') or 0)>=0
    st.markdown(f'''<div class="detail-hero"><img src="{escape(asset['logo'],quote=True)}" alt="{escape(asset['name'])} logo"/>
        <div><h1>{escape(asset['name'])}</h1><p>{ticker} · {'Crypto' if asset['kind']=='crypto' else 'Stock'}</p></div></div>
        <div class="detail-price">{money(asset['current_price'])}</div>
        <div class="detail-change {'gain' if positive else 'loss'}">{signed_money(quote.get('change_24h'))} ({quote.get('change_24h_pct',0):+.2f}%) <span class="muted">· {'Past 24h' if quote.get('change_basis')=='rolling_24h' else 'vs. previous close'}</span></div>''',unsafe_allow_html=True)
    period_default={} if 'timeframe' in st.session_state else {'default':st.session_state.chart_timeframe}
    st.pills('Chart period',['24H','7D','30D','90D'],key='timeframe',selection_mode='single',required=True,on_change=sync_timeframe,label_visibility='collapsed',**period_default)
    st.plotly_chart(chart_plotter.plot_asset_history(asset,st.session_state.timeframe),width='stretch',config={'displayModeBar':False,'scrollZoom':False})
    st.markdown('**Check Past Market Value**')
    slider,exact=st.columns([4,1])
    slider_default={} if 'lookback_slider' in st.session_state else {'value':st.session_state.lookback_days}
    exact_default={} if 'lookback_exact' in st.session_state else {'value':st.session_state.lookback_days}
    with slider: st.slider('Days ago',1,90,key='lookback_slider',on_change=sync_lookback,args=('lookback_slider',),**slider_default)
    with exact: st.number_input('Exact days ago',1,90,key='lookback_exact',step=1,on_change=sync_lookback,args=('lookback_exact',),**exact_default)
    date=local_timestamp(asset['historical_timestamp'])
    direction='gain' if asset['change']>=0 else 'loss'
    st.markdown(f'''<div class="time-machine"><div class="eyebrow">{asset['days_ago']} days ago → today</div>
        <div class="comparison">{money(asset['historical_price'])} <span class="muted">→</span> {money(asset['current_price'])}
        <span class="{direction}">({asset['change_pct']:+.2f}%)</span></div>
        <small>Requested: {escape(local_timestamp(asset['requested_timestamp']))}<br/>Market observation: {escape(date)} · {asset['interval']} bar</small></div>''',unsafe_allow_html=True)
    caption('The marker shows the last completed market bar at or before your selected time. Closed-market dates use the prior observation.')
    buy,sell=st.columns(2)
    with buy:
        if st.button('BUY',key='buy_action',width='stretch'):
            st.session_state.order_action='BUY';st.session_state.order_size=0.;st.rerun()
    with sell:
        if st.button('SELL',key='sell_action',width='stretch'):
            st.session_state.order_action='SELL';st.session_state.order_size=0.;st.rerun()
    render_order(asset)
    if asset.get('holding'):
        h=asset['holding']
        caption(f"Your position · {h['quantity']:,.6f} {ticker} · Value {money(h['market_value'])} · P&L {signed_money(h['unrealized_pnl'])}")
    caption(f"Latest quote: {local_timestamp(quote['timestamp'])} · {quote.get('market_status','unknown').replace('_',' ')} · Prices may be delayed")


def render_history(wallet):
    st.button('← Back to Wallet',type='tertiary',on_click=go_home)
    st.subheader('Your trading history')
    st.plotly_chart(chart_plotter.plot_wallet_history(wallet['snapshots'],wallet['starting_cash']),width='stretch',config={'displayModeBar':False})
    performance,details=performance_text(wallet)
    caption(f"{wallet['total_trades']} trades · {performance} · {details}")
    if wallet['trade_ledger']:
        rows=[{'Time (IST)':local_timestamp(r['timestamp']),'Asset':r['ticker'],'Action':r['action'],
               'Quantity':f"{r['quantity']:,.8f}",'Fill price':money(r['price']),
               'Realized P&L':signed_money(r['realized_pnl']),'Fill source':r['execution_label']}
              for r in wallet['trade_ledger']]
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    else: st.info('Your completed trades will appear here.')
    with st.expander('Position protection',expanded=False):
        caption('Refreshes update trailing peaks. Run this check to execute any active stop-loss or take-profit rules.')
        if st.button('Check protection rules',width='stretch'):
            result=api('POST','/api/wallet/risk-check')
            st.session_state.flash=f"Protection checked · {result['triggered_count']} position(s) closed."
            st.rerun()


def trade_prediction(ticker):
    st.session_state.main_navigation='👛 Wallet'
    open_asset(ticker,'BUY')


def clear_prediction():
    st.session_state.pop('prediction_cache',None)


def render_academic_metrics(payload):
    metrics=payload.get('metrics',{})
    def metric_number(name,suffix=''):
        value=metrics.get(name)
        return '—' if value is None else f'{float(value):,.2f}{suffix}'
    cols=st.columns(3)
    for col,label,key,suffix in zip(cols,['Test RMSE (USD)','Test MAE (USD)','Directional accuracy'],
            ['rmse','mae','directional_accuracy'],['','','%']):
        with col:st.metric(label,metric_number(key,suffix))
    caption(f"Naive baseline · RMSE {metric_number('naive_rmse')} USD · MAE {metric_number('naive_mae')} USD · Directional accuracy {metric_number('naive_directional_accuracy','%')}")
    if 'beats_baseline' in metrics:
        caption('LSTM has lower test RMSE than the persistence baseline.' if metrics['beats_baseline'] else 'LSTM does not beat the persistence baseline on test RMSE.')
    caption(f"Chronological 80/20 split · Training-only scaling · {metrics.get('test_samples','—')} held-out origins · 30 stochastic passes · No guarantee of future accuracy.")


def render_academic_inspector(ticker,horizon,payload):
    with st.expander('🎓 Academic & Quantitative Inspector (RMSE / Raw Indicators / CSV Batch)',expanded=False):
        if payload:
            render_academic_metrics(payload)
            st.download_button('Download predictions CSV',academic.predictions_csv(payload),
                file_name=f'capitex_{ticker}_{horizon}_predictions.csv',mime='text/csv',key='download_predictions')
        else:caption('Market evaluation is unavailable. You can still inspect an uploaded dataset below.')
        try:
            ledger=api('GET','/api/wallet/ledger')['rows']
            st.download_button('Download trade ledger CSV',academic.ledger_csv(ledger),
                file_name='capitex_trade_ledger.csv',mime='text/csv',key='download_ledger')
        except (RuntimeError,KeyError):caption('Trade ledger export is temporarily unavailable.')
        st.markdown('**Raw technical inspection**')
        caption('Candlesticks, SMA 20/50, EMA 12/26, Wilder RSI 14 and MACD 12/26/9. The latest 252 observations are shown.')
        if st.button('Load technical chart',key='load_technical'):
            try:
                with st.spinner('Loading technical history…'):
                    st.session_state.technical_inspection=(ticker,api('GET',f'/api/inspect/{ticker}'))
            except RuntimeError as exc:st.error(str(exc))
        technical=st.session_state.get('technical_inspection')
        if technical and technical[0]==ticker:
            records=technical[1]['history']
            st.plotly_chart(chart_plotter.plot_academic_chart(records,ticker),key='academic_technical_chart',width='stretch')
            st.dataframe(academic.records_frame(records).tail(30),width='stretch')
        st.markdown('**CSV dataset inspection**')
        caption('Upload daily OHLCV data: Date (or Timestamp), Open, High, Low, Close, Volume. Use at least 400 rows; maximum 2 MB / 10,000 rows. Times without a timezone are treated as UTC.')
        uploaded=st.file_uploader('Historical CSV dataset',type=['csv'],max_upload_size=2,key='academic_csv')
        if st.button('Analyze CSV dataset',key='analyze_csv',disabled=uploaded is None):
            try:
                content=uploaded.getvalue()
                if len(content)>2_000_000:raise ValueError('CSV file must be no larger than 2 MB.')
                with st.spinner('Training and evaluating your dataset… This may take a minute.'):
                    result=api('POST','/api/inspect/csv',{'csv_text':content.decode('utf-8-sig'),'horizon':horizon})
                st.session_state.csv_inspection=result
            except (RuntimeError,ValueError,UnicodeError) as exc:st.error(str(exc))
        result=st.session_state.get('csv_inspection')
        if result:
            st.info(f"CSV simulation · {result['rows']} cleaned daily rows · Anchored to the dataset’s last close. This is separate from live forecasts and cannot place trades.")
            csv_forecast=result['forecast']
            caption(f"Dataset target horizon: {csv_forecast['horizon_label']}")
            render_academic_metrics(csv_forecast)
            st.plotly_chart(chart_plotter.plot_prediction_corridor(csv_forecast),key='csv_corridor',width='stretch')
            st.plotly_chart(chart_plotter.plot_academic_chart(result['technical_history'],'Uploaded CSV'),key='csv_technical_chart',width='stretch')
            st.download_button('Download CSV dataset predictions',academic.predictions_csv(csv_forecast),
                file_name='capitex_uploaded_predictions.csv',mime='text/csv',key='download_csv_predictions')
            st.dataframe(pd.DataFrame([result['cleaning_report']]),hide_index=True,width='stretch')


def render_reasoning(why):
    sentiment=why.get('sentiment',{})
    score=max(0,min(100,float(sentiment.get('score',50))))
    label=escape(sentiment.get('label','Neutral'))
    st.markdown(f'''<div class="reason-block"><div class="reason-title">Public & market sentiment</div>
        <div class="sentiment-reading">{score:.0f}<span> / 100 · {label}</span></div>
        <div class="sentiment-track" role="meter" aria-label="Fear and greed sentiment" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{score}"><i style="left:clamp(7px,{score}%,calc(100% - 7px))"></i></div>
        <div class="sentiment-labels"><span>Fear</span><span>Neutral</span><span>Greed</span></div></div>''',unsafe_allow_html=True)
    caption(f"CAPITEX sentiment blend · {sentiment.get('headlines_used',0)} recent headlines included. Missing sources are omitted.")
    for heading,tags,empty in [
        ('Technical summary',why.get('technical_summary',{}).get('tags',[]),'Technical readings are temporarily unavailable.'),
        ('Catalysts to watch',why.get('catalyst_tags',[]),'No verified near-term catalysts found.')]:
        st.markdown(f'<div class="reason-title">{heading}</div>',unsafe_allow_html=True)
        if tags:
            badges=[]
            for tag in tags:
                tone=tag.get('tone','neutral')
                tone=tone if tone in ('bullish','bearish','neutral','caution') else 'neutral'
                badges.append(f'<span class="factor-pill {tone}">{escape(tag["label"])}</span>')
            st.markdown('<div class="factor-pills">'+''.join(badges)+'</div>',unsafe_allow_html=True)
        else: caption(empty)
    caption('These factors describe market context; they are not explanations of the LSTM’s individual decisions.')


def render_predictor():
    st.markdown('<div class="predictor-title"><div class="eyebrow">CAPITEX AI</div><h1>A clearer view of what’s next.</h1><p>One outlook. A range of possibilities.</p></div>',unsafe_allow_html=True)
    chooser,refresh=st.columns([4,1],vertical_alignment='bottom')
    with chooser:
        ticker=st.selectbox('Choose a market',config.DEFAULT_TICKERS,key='prediction_asset',
            format_func=lambda t:f'{config.ASSET_UNIVERSE[t]["name"]} · {t}')
    with refresh:
        st.button('Refresh outlook',key='refresh_prediction',width='stretch',on_click=clear_prediction)
    # Keep the selector above the slow request and retain it when a source is unavailable.
    horizon_labels=['Next 24 Hours','7 Days','30 Days']
    saved=st.session_state.get('prediction_horizon','7 Days')
    left,middle,right=st.columns([1.2,1,1.5],gap='small')
    with right:
        with st.container(key='horizon_card',border=True):
            st.markdown('<div class="eyebrow">Target horizon</div>',unsafe_allow_html=True)
            default={} if 'prediction_horizon' in st.session_state else {'default':saved}
            selected=st.segmented_control('Forecast horizon',horizon_labels,key='prediction_horizon',
                required=True,width='stretch',label_visibility='collapsed',**default)
            crypto=config.ASSET_UNIVERSE[ticker]['kind']=='crypto'
            caption('Calendar days · markets trade 24/7' if crypto else 'Trading sessions · exchange holidays may shift dates')
    horizon={'Next 24 Hours':'24h','7 Days':'7d','30 Days':'30d'}[selected]
    cache=st.session_state.get('prediction_cache',{})
    key=(ticker,horizon)
    payload=None
    try:
        if key not in cache or monotonic()-cache[key][0]>300:
            with st.spinner('Building your outlook… The first forecast may take a minute.'):
                forecast=api('GET',f'/api/predict/{ticker}?horizon={horizon}')
            cache[key]=(monotonic(),forecast)
            st.session_state.prediction_cache=cache
        payload=cache[key][1]
        hero=payload['hero']
        outlook=hero['outlook']
        outlook=outlook if outlook in ('bullish','bearish','neutral') else 'neutral'
        title={'bullish':'▲ Bullish Outlook','bearish':'▼ Bearish Outlook','neutral':'◆ Neutral Outlook'}[outlook]
        with left:
            st.markdown(f'''<div class="outlook-card {outlook}"><div class="eyebrow">Expected direction</div>
                <div class="outlook-badge">{title}</div><div class="outlook-move">{float(hero['expected_change_pct']):+.2f}%</div>
                <div class="card-note">Median target <b>{money(hero['median_target_price'])}</b></div></div>''',unsafe_allow_html=True)
        confidence=max(0,min(99,float(hero['confidence'])))
        with middle:
            st.markdown(f'''<div class="confidence-card"><div class="eyebrow">Model certainty</div>
                <div class="confidence-ring" role="img" aria-label="{confidence:.0f}% model certainty"><svg viewBox="0 0 120 120" aria-hidden="true"><circle cx="60" cy="60" r="48" fill="none" stroke="#302A40" stroke-width="7"/><circle cx="60" cy="60" r="48" fill="none" stroke="#AB9FF2" stroke-width="7" stroke-linecap="round" pathLength="100" stroke-dasharray="{confidence} 100" transform="rotate(-90 60 60)"/></svg><div><b>{confidence:.0f}%</b><span>Certainty</span></div></div></div>''',unsafe_allow_html=True)
        quote=payload['quote']
        name=config.ASSET_UNIVERSE[ticker]['name']
        st.markdown(f'''<div class="forecast-heading"><div><h3>{escape(name)} <span class="forecast-ticker">{escape(ticker)}</span></h3><span class="chart-key">● Recent price <i>● Estimated 80% range</i></span></div><b>{money(payload['current_price'])}<small>Current quote</small></b></div>''',unsafe_allow_html=True)
        if quote.get('is_fallback') or quote.get('is_stale'):
            st.warning('Using the latest available historical or delayed quote. This outlook may not reflect current trading.')
        st.plotly_chart(chart_plotter.plot_prediction_corridor(payload),key='prediction_corridor',width='stretch',config={'displayModeBar':False,'scrollZoom':False})
        caption(f"As of {local_timestamp(payload.get('forecast_as_of',quote['timestamp']))} · Estimated 80% range · Certainty is a model score, not a success probability.")
        with st.expander('🔍 View AI Reasoning Factors',expanded=False):
            render_reasoning(payload['why'])
        st.button(f'Trade {ticker} in CAPITEX Wallet →',key='trade_prediction',width='stretch',
            type='primary',on_click=trade_prediction,args=(ticker,))
    except (RuntimeError,ValueError,KeyError) as exc:
        st.error(str(exc) if isinstance(exc,RuntimeError) else 'The outlook is incomplete. Please refresh it in a moment.')
        caption('Choose another market or use Refresh outlook to try again.')
    render_academic_inspector(ticker,horizon,payload)


for key,value in {'view':'home','asset':'AAPL','order_action':None,'lookback_days':44,
                  'chart_timeframe':'90D'}.items():
    if key not in st.session_state: st.session_state[key]=value
logo=branding._data_uri(config.LOGO_PATH)
st.markdown(f'<div class="brand"><div class="brand-name"><img src="{logo}" alt="CAPITEX logo"/>CAPITEX</div><span class="simulated"><i></i>VIRTUAL TRADING</span></div>',unsafe_allow_html=True)
navigation_default={} if 'main_navigation' in st.session_state else {'default':'👛 Wallet'}
nav=st.segmented_control('CAPITEX navigation',['👛 Wallet','🧠 AI Predictor'],required=True,
                         label_visibility='collapsed',width='stretch',key='main_navigation',**navigation_default)


@st.fragment(run_every=30)
def wallet_screen():
    if st.session_state.get('flash'):
        st.success(st.session_state.pop('flash'))
    try:
        if st.session_state.view=='asset':
            render_asset()
            return
        wallet=api('GET','/api/wallet')
        if st.session_state.view=='history': render_history(wallet)
        elif st.session_state.view=='choose':
            st.button('← Back to Wallet',type='tertiary',on_click=go_home)
            action=st.session_state.order_action
            st.subheader('Choose an asset to '+action.lower())
            choices=wallet['holdings'] if action=='SELL' else wallet['watchlist']
            if not choices: st.info('You do not have any positions to sell yet. Explore markets to get started.')
            for asset in choices: asset_card(asset,'choose',action=='SELL',action)
        else:
            with st.container(horizontal=True,horizontal_alignment='right'):
                st.button('Refresh',icon=':material/refresh:',key='refresh_prices',type='tertiary',help='Update prices and portfolio')
            render_home(wallet)
    except RuntimeError as exc:
        st.error(str(exc))
        st.button('Try again',key='retry_wallet')


if nav=='👛 Wallet': wallet_screen()
else:
    render_predictor()
st.markdown(f'<div class="footer">{escape(config.FOOTER_TEXT)}</div>',unsafe_allow_html=True)
