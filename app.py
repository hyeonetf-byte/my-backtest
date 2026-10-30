import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 웹페이지 기본 설정
st.set_page_config(page_title="나만의 ETF 백테스트", layout="wide")
st.title("📈 ETF 포트폴리오 백테스트 시뮬레이터")
st.markdown("**조건:** 배당금은 현금 누적 후 매년 4월 말, 10월 말 리밸런싱 시점에만 재투자 (JEPQ 상장일 기준)")

tickers = ['QLD', 'SCHD', 'DIVO', 'JEPI', 'JEPQ']

# 데이터 다운로드 (캐싱을 사용하여 매번 다운로드하지 않음)
@st.cache_data
def load_data():
    dfs = {}
    for t in tickers:
        ticker_obj = yf.Ticker(t)
        df_t = ticker_obj.history(period="5y", auto_adjust=False)
        dfs[t] = df_t[['Close', 'Dividends']]
    
    close_df = pd.DataFrame({t: dfs[t]['Close'] for t in tickers}).dropna()
    div_df = pd.DataFrame({t: dfs[t]['Dividends'] for t in tickers}).reindex(close_df.index).fillna(0)
    
    # 4월, 10월 마지막 거래일 찾기
    months = close_df.index.month
    years = close_df.index.year
    monthly_last_days = close_df.groupby([years, months]).apply(lambda x: x.index.max())
    
    rebalance_dates = set()
    for (y, m), d in monthly_last_days.items():
        if m in [4, 10]:
            rebalance_dates.add(d)
            
    return close_df, div_df, rebalance_dates

# 로딩 안내
with st.spinner('데이터를 불러오는 중입니다... 잠시만 기다려주세요.'):
    close_df, div_df, rebalance_dates = load_data()

# 사이드바 (설정창)
st.sidebar.header("📊 비중 설정 (%)")
w_qld = st.sidebar.slider("QLD", 0, 100, 20, 5)
w_schd = st.sidebar.slider("SCHD", 0, 100, 20, 5)
w_divo = st.sidebar.slider("DIVO", 0, 100, 20, 5)
w_jepi = st.sidebar.slider("JEPI", 0, 100, 20, 5)
w_jepq = st.sidebar.slider("JEPQ", 0, 100, 20, 5)

weights = [w_qld, w_schd, w_divo, w_jepi, w_jepq]
total_w = sum(weights)

if total_w == 0:
    st.error("⚠️ 비중의 합이 0입니다. 왼쪽 사이드바에서 비중을 조절해주세요.")
else:
    # 비중 정규화 (자동으로 100% 기준으로 맞춤)
    weights_dict = {t: w / total_w for t, w in zip(tickers, weights)}
    
    # 백테스트 로직
    initial_capital = 10000.0
    cash = 0.0
    shares = {t: 0.0 for t in tickers}
    portfolio_values = []
    
    dates = close_df.index
    for t in tickers:
        shares[t] = (initial_capital * weights_dict[t]) / close_df.loc[dates[0], t]
        
    for current_date in dates:
        # 배당금 누적
        for t in tickers:
            div = div_df.loc[current_date, t]
            if div > 0:
                cash += shares[t] * div
                
        # 리밸런싱 (4월 말, 10월 말)
        if current_date in rebalance_dates:
            current_val = sum(shares[t] * close_df.loc[current_date, t] for t in tickers) + cash
            for t in tickers:
                shares[t] = (current_val * weights_dict[t]) / close_df.loc[current_date, t]
            cash = 0.0 
            
        daily_val = sum(shares[t] * close_df.loc[current_date, t] for t in tickers) + cash
        portfolio_values.append(daily_val)
        
    port_series = pd.Series(portfolio_values, index=dates)
    daily_returns = port_series.pct_change().dropna()
    
    # 지표 계산
    years_passed = (dates[-1] - dates[0]).days / 365.25
    cagr = ((port_series.iloc[-1] / port_series.iloc[0]) ** (1 / years_passed) - 1) * 100
    
    cum_returns = port_series / port_series.iloc[0]
    mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min() * 100
    
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    sortino = daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252)

    # 화면 상단 결과 요약
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("연평균 수익률 (CAGR)", f"{cagr:.2f}%")
    col2.metric("최대 낙폭 (MDD)", f"{mdd:.2f}%")
    col3.metric("샤프 지수 (Sharpe)", f"{sharpe:.2f}")
    col4.metric("최종 자산 ($10,000 투자 시)", f"${port_series.iloc[-1]:,.0f}")

    # 인터랙티브 차트 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_series.index, y=port_series.values, mode='lines', name='내 포트폴리오'))
    fig.update_layout(title="포트폴리오 자산 성장 추이", yaxis_title="자산 규모 ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # 상관관계 표
    st.subheader("🔗 ETF 간 상관관계 (일간 수익률 기준)")
    corr_matrix = close_df.pct_change().corr().round(2)
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None))
