import streamlit as st
import pandas as pd
import io
from data_manager import fetch_and_store_data, get_matches_df, calculate_standings, get_next_fixtures, get_top_scorers, get_live_matches
from predictor import MatchPredictor
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Football Analyst 2026", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .prediction-card {
        background-color: #1f2937; padding: 15px; border-radius: 10px;
        border: 1px solid #374151; margin-bottom: 10px; text-align: center;
    }
    .score-badge {
        background-color: #374151; color: #fbbf24; padding: 5px 10px;
        border-radius: 5px; font-weight: bold; font-size: 18px; margin: 2px;
    }
    .stat-box {
        background-color: #111827; padding: 10px; border-radius: 8px;
        border-left: 4px solid #3b82f6; margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- YARDIMCI FONKSİYONLAR ---
def calculate_performance(df, team_name, is_home):
    if is_home:
        matches = df[df['home_team'] == team_name]
        total = len(matches)
        if total == 0: return 0, 0, 0, 0
        wins = len(matches[matches['home_score'] > matches['away_score']])
        draws = len(matches[matches['home_score'] == matches['away_score']])
        losses = len(matches[matches['home_score'] < matches['away_score']])
    else:
        matches = df[df['away_team'] == team_name]
        total = len(matches)
        if total == 0: return 0, 0, 0, 0
        wins = len(matches[matches['away_score'] > matches['home_score']])
        draws = len(matches[matches['away_score'] == matches['home_score']])
        losses = len(matches[matches['away_score'] < matches['home_score']])
    return total, (wins/total)*100, (draws/total)*100, (losses/total)*100

def create_excel_download(result_data, home, away):
    data = {
        'Tarih': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
        'Ev Sahibi': [home], 'Deplasman': [away],
        'MS 1 (%)': [f"%{result_data['prob_home']:.1f}"],
        'MS X (%)': [f"%{result_data['prob_draw']:.1f}"],
        'MS 2 (%)': [f"%{result_data['prob_away']:.1f}"],
        '2.5 Üst (%)': [f"%{result_data['prob_over_2_5']:.1f}"],
        'KG Var (%)': [f"%{result_data['prob_btts_yes']:.1f}"],
        'Güven Skoru (%)': [f"%{result_data.get('confidence', 0):.1f}"],
        'Skor Tahmini': [result_data['top_scores'][0]['score'] if len(result_data['top_scores']) > 0 else '-']
    }
    df_export = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Tahmin')
    return output.getvalue()

def get_last_5_goals(df, team_name):
    """Takımın son 5 maçındaki gollerini getirir."""
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].sort_values('date', ascending=False).head(5)
    team_matches = team_matches.sort_values('date', ascending=True)
    
    goals = []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team_name:
            goals.append(row['home_score'])
        else:
            goals.append(row['away_score'])
    return goals

def get_last_5_btts(df, team_name):
    """Takımın son 5 maçındaki KG Var/Yok istatistiğini getirir."""
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].sort_values('date', ascending=False).head(5)
    
    yes = 0
    no = 0
    for _, row in team_matches.iterrows():
        if row['home_score'] > 0 and row['away_score'] > 0:
            yes += 1
        else:
            no += 1
    return yes, no

def get_home_away_avg_goals(df, team_name):
    """Takımın iç saha ve dış saha gol ortalamalarını hesaplar."""
    home_matches = df[df['home_team'] == team_name]
    away_matches = df[df['away_team'] == team_name]
    
    avg_home_goals_scored = home_matches['home_score'].mean() if not home_matches.empty else 0
    avg_away_goals_scored = away_matches['away_score'].mean() if not away_matches.empty else 0
    
    return avg_home_goals_scored, avg_away_goals_scored

def get_last_5_ht_stats(df, team_name):
    """Takımın son 5 maçındaki İlk Yarı sonuçlarını (G/B/M) sayar."""
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)].sort_values('date', ascending=False).head(5)
    
    wins = 0
    draws = 0
    losses = 0
    
    for _, row in team_matches.iterrows():
        ht_home_score = row['ht_home_score']
        ht_away_score = row['ht_away_score']
        
        if pd.isna(ht_home_score) or pd.isna(ht_away_score):
            continue

        if row['home_team'] == team_name: # Team is home
            if ht_home_score > ht_away_score: wins += 1
            elif ht_home_score == ht_away_score: draws += 1
            else: losses += 1
        else: # Team is away
            if ht_away_score > ht_home_score: wins += 1
            elif ht_away_score == ht_home_score: draws += 1
            else: losses += 1
            
    return wins, draws, losses

def get_points_progression(df, team_name, competition):
    """Takımın sezonluk puan değişimini hesaplar."""
    comp_df = df[df['competition'] == competition]
    if comp_df.empty: return []
    
    latest_season = comp_df['season'].max()
    season_df = comp_df[comp_df['season'] == latest_season]
    
    team_matches = season_df[(season_df['home_team'] == team_name) | (season_df['away_team'] == team_name)].sort_values('date')
    
    points = 0
    progression = []
    
    for _, row in team_matches.iterrows():
        if row['status'] != 'FINISHED': continue
        
        if row['home_team'] == team_name:
            if row['home_score'] > row['away_score']: points += 3
            elif row['home_score'] == row['away_score']: points += 1
        else:
            if row['away_score'] > row['home_score']: points += 3
            elif row['away_score'] == row['home_score']: points += 1
            
        progression.append(points)
        
    return progression

# --- ANA UYGULAMA ---
st.title("AI Football Analyst 2026")

with st.sidebar:
    st.header("Kontrol Paneli")
    
    user_api_key = st.text_input("🔑 API Anahtarı (Opsiyonel)", type="password", help="Kendi API anahtarınızı kullanmak isterseniz buraya girin.")
    
    if st.button("Verileri Güncelle (API)"):
        with st.spinner("Veriler işleniyor... (Bu işlem 1-2 dk sürebilir)"):
            fetch_and_store_data(api_key=user_api_key if user_api_key else None, seasons=[2023, 2024, 2025, 2026])
        st.success("Güncelleme Tamamlandı!")
        st.rerun()
        
    st.markdown("---")
    st.header("Canlı Maçlar")
    if st.button("Canlı Skorları Yenile"):
        live_games = get_live_matches(user_api_key if user_api_key else None)
        st.session_state['live_games'] = live_games
    
    if 'live_games' in st.session_state:
        if st.session_state['live_games']:
            for game in st.session_state['live_games']:
                st.markdown(f"**{game['lig']}**: {game['ev']} {game['skor']} {game['deplasman']}")
                st.markdown("---")
        else:
            st.caption("Şu an takip edilen liglerde canlı maç yok.")

df = get_matches_df()

if df.empty:
    st.warning("Veritabanı boş. Lütfen sol menüden 'Verileri Güncelle' butonuna basın.")
else:
    predictor = MatchPredictor(df)
    is_trained = predictor.train()
    
    # Model Başarısını Sidebar'da Göster
    if is_trained:
        st.sidebar.metric(label="Model Doğruluk Oranı", value=f"%{predictor.accuracy:.1f}", delta="Test Verisi")
    
    if is_trained:
        # --- LİG SEÇİMİ ---
        leagues = df['competition'].unique()
        sel_league = st.selectbox("Lig Seçiniz", leagues)
        
        # --- FİKSTÜR VE PUAN DURUMU ---
        col_fix, col_stand = st.columns(2)
        with col_fix:
            with st.expander(f"{sel_league} - Gelecek Maçlar", expanded=True):
                fixtures = get_next_fixtures(sel_league)
                if not fixtures.empty:
                    for idx, row in fixtures.iterrows():
                        match_date = pd.to_datetime(row['date']).strftime('%d.%m %H:%M')
                        h_team, a_team = row['home_team'], row['away_team']
                        c1, c2 = st.columns([3, 1])
                        c1.write(f"**{match_date}**: {h_team} - {a_team}")
                        if c2.button("Seç", key=f"btn_{idx}"):
                            st.session_state['sb_home'] = h_team
                            st.session_state['sb_away'] = a_team
                            st.rerun()
                else:
                    st.info("Yakın tarihte maç yok.")

        with col_stand:
            with st.expander(f"{sel_league} - Puan Durumu", expanded=False):
                standings = calculate_standings(df, sel_league)
                if not standings.empty:
                    st.dataframe(standings.style.background_gradient(subset=['P'], cmap="Greens"), width='stretch', height=300)
                else:
                    st.info("Puan durumu yok.")

            with st.expander(f"{sel_league} - Gol Krallığı", expanded=False):
                scorers = get_top_scorers(sel_league)
                if not scorers.empty:
                    st.dataframe(scorers, width='stretch', height=300)
                else:
                    st.info("Gol krallığı verisi bulunamadı (Verileri güncellemeyi deneyin).")

        st.markdown("---")

        # --- SÜRPRİZ MAÇ ARAYICI ---
        with st.expander("Sürpriz Maç Arayıcı (Yüksek Potansiyel)", expanded=False):
            st.info("Bu araç, seçili ligdeki gelecek maçları tarar ve **Deplasman Galibiyeti (%40+)** veya **Yüksek Beraberlik (%35+)** ihtimali olan maçları listeler.")
            
            if st.button("🔍 Sürprizleri Tara"):
                with st.spinner("Fikstür taranıyor ve analiz ediliyor..."):
                    upcoming = get_next_fixtures(sel_league, limit=20)
                    surprises = []
                    
                    if not upcoming.empty:
                        for _, row in upcoming.iterrows():
                            h, a = row['home_team'], row['away_team']
                            date_str = pd.to_datetime(row['date']).strftime('%d.%m %H:%M')
                            
                            # Tahmin yap
                            pred = predictor.predict_match(h, a)
                            if pred:
                                reasons = []
                                if pred['prob_away'] >= 40:
                                    reasons.append(f"Deplasman Şansı: %{pred['prob_away']:.1f}")
                                if pred['prob_draw'] >= 35:
                                    reasons.append(f"Beraberlik Şansı: %{pred['prob_draw']:.1f}")
                                
                                if reasons:
                                    surprises.append({'date': date_str, 'home': h, 'away': a, 'reasons': reasons})
                    
                    st.session_state['surprise_results'] = surprises
            
            if 'surprise_results' in st.session_state and st.session_state['surprise_results']:
                st.success(f"{len(st.session_state['surprise_results'])} adet potansiyel sürpriz maç bulundu!")
                for s in st.session_state['surprise_results']:
                    c1, c2 = st.columns([3, 1])
                    c1.write(f"**{s['date']}**: {s['home']} - {s['away']} ({', '.join(s['reasons'])})")
                    if c2.button("Analiz Et", key=f"btn_surp_{s['home']}_{s['away']}"):
                        st.session_state['sb_home'] = s['home']
                        st.session_state['sb_away'] = s['away']
                        st.rerun()
                
                if st.button("Listeyi Temizle"):
                    del st.session_state['surprise_results']
                    st.rerun()
            elif 'surprise_results' in st.session_state:
                st.warning("Kriterlere uyan sürpriz maç bulunamadı.")

        st.markdown("---")

        # --- TAKIM SEÇİMİ ---
        teams = df[df['competition'] == sel_league]['home_team'].unique()
        teams.sort()
        
        idx_home = list(teams).index(st.session_state['selected_home']) if 'selected_home' in st.session_state and st.session_state['selected_home'] in teams else 0
        idx_away = list(teams).index(st.session_state['selected_away']) if 'selected_away' in st.session_state and st.session_state['selected_away'] in teams else (1 if len(teams)>1 else 0)

        col1, col2 = st.columns(2)
        with col1: home = st.selectbox("Ev Sahibi", teams, index=idx_home, key='sb_home')
        with col2: away = st.selectbox("Deplasman", teams, index=idx_away, key='sb_away')

        # --- PERFORMANS ---
        st.markdown("### İç Saha vs Dış Saha Performansı")
        h_total, h_win, h_draw, h_loss = calculate_performance(df, home, is_home=True)
        a_total, a_win, a_draw, a_loss = calculate_performance(df, away, is_home=False)
        
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(f"<div class='stat-box'><b>{home}</b> (İç Saha - {h_total} Maç)</div>", unsafe_allow_html=True)
            st.progress(int(h_win))
            st.caption(f"G: %{h_win:.1f} | B: %{h_draw:.1f} | M: %{h_loss:.1f}")
        with p2:
            st.markdown(f"<div class='stat-box'><b>{away}</b> (Deplasman - {a_total} Maç)</div>", unsafe_allow_html=True)
            st.progress(int(a_win))
            st.caption(f"G: %{a_win:.1f} | B: %{a_draw:.1f} | M: %{a_loss:.1f}")

        st.markdown("---")

        # --- HEAD TO HEAD (ARALARINDAKİ MAÇLAR) ---
        st.markdown("### Aralarındaki Maçlar (H2H)")
        h2h_matches = df[((df['home_team'] == home) & (df['away_team'] == away)) | 
                         ((df['home_team'] == away) & (df['away_team'] == home))].sort_values('date', ascending=False).head(5)
        
        if not h2h_matches.empty:
            last_match = h2h_matches.iloc[0]
            lm_date = pd.to_datetime(last_match['date']).strftime('%d.%m.%Y')
            st.info(f"**Son Karşılaşma:** {last_match['home_team']} **{last_match['home_score']} - {last_match['away_score']}** {last_match['away_team']} ({lm_date})")
            
            for _, row in h2h_matches.iterrows():
                d = pd.to_datetime(row['date']).strftime('%d.%m.%Y')
                h, a = row['home_team'], row['away_team']
                hs, as_ = row['home_score'], row['away_score']
                
                # Renklendirme
                color = "#ffffff"
                if (row['home_team'] == home and hs > as_) or (row['away_team'] == home and as_ > hs): color = "#10b981" # Kazanma
                elif hs == as_: color = "#9ca3af" # Beraberlik
                else: color = "#ef4444" # Kaybetme
                
                st.markdown(f"<div style='background-color:#1f2937; padding:8px; margin:2px; border-radius:5px; border-left:5px solid {color}'>"
                            f"<small>{d}</small> <b>{h} {hs} - {as_} {a}</b></div>", unsafe_allow_html=True)
        else:
            st.info("Bu iki takım arasında kayıtlı geçmiş maç bulunamadı.")

        st.markdown("---")

        # --- SON 5 MAÇ GOL GRAFİĞİ ---
        st.markdown("### Son 5 Maç Gol İstatistikleri")
        goals_h = get_last_5_goals(df, home)
        goals_a = get_last_5_goals(df, away)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write(f"**{home}**")
            st.write(f"Goller: {goals_h}")
            st.write(f"Ortalama: {sum(goals_h)/len(goals_h) if goals_h else 0:.2f}")
        with col_g2:
            st.write(f"**{away}**")
            st.write(f"Goller: {goals_a}")
            st.write(f"Ortalama: {sum(goals_a)/len(goals_a) if goals_a else 0:.2f}")

        st.markdown("---")

        # --- SON 5 MAÇ KG VAR/YOK PASTA GRAFİĞİ ---
        st.markdown("### Son 5 Maç KG Var/Yok")
        btts_h_yes, btts_h_no = get_last_5_btts(df, home)
        btts_a_yes, btts_a_no = get_last_5_btts(df, away)

        col_pie1, col_pie2 = st.columns(2)
        
        # Home Pie
        with col_pie1:
            st.write(f"**{home}**")
            if btts_h_yes + btts_h_no > 0:
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                ax1.pie([btts_h_yes, btts_h_no], labels=['Var', 'Yok'], autopct='%1.1f%%', colors=['#10b981', '#ef4444'], textprops={'color':"white"})
                fig1.patch.set_facecolor('#0e1117')
                st.pyplot(fig1, width='content')

        # Away Pie
        with col_pie2:
            st.write(f"**{away}**")
            if btts_a_yes + btts_a_no > 0:
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                ax2.pie([btts_a_yes, btts_a_no], labels=['Var', 'Yok'], autopct='%1.1f%%', colors=['#10b981', '#ef4444'], textprops={'color':"white"})
                fig2.patch.set_facecolor('#0e1117')
                st.pyplot(fig2, width='content')

        st.markdown("---")

        # --- İÇ SAHA/DIŞ SAHA GOL ORTALAMALARI ---
        st.markdown("### İç Saha / Dış Saha Gol Ortalamaları")
        h_avg_home, h_avg_away = get_home_away_avg_goals(df, home)
        a_avg_home, a_avg_away = get_home_away_avg_goals(df, away)

        col_avg1, col_avg2 = st.columns(2)
        col_avg1.metric(f"{home} (İç Saha)", f"{h_avg_home:.2f} Gol")
        col_avg2.metric(f"{away} (Dış Saha)", f"{a_avg_away:.2f} Gol")

        st.markdown("---")

        # --- SEZONLUK PERFORMANS TRENDİ ---
        st.markdown("### Sezonluk Performans (Puan)")
        trend_h = get_points_progression(df, home, sel_league)
        trend_a = get_points_progression(df, away, sel_league)
        
        col_p1, col_p2 = st.columns(2)
        if trend_h:
            col_p1.metric(f"{home} Toplam Puan", trend_h[-1])
        if trend_a:
            col_p2.metric(f"{away} Toplam Puan", trend_a[-1])

        st.markdown("---")

        # --- SON 5 MAÇ İLK YARI SONUÇLARI ---
        st.markdown("### Son 5 Maç İlk Yarı Sonuçları")
        ht_h_w, ht_h_d, ht_h_l = get_last_5_ht_stats(df, home)
        ht_a_w, ht_a_d, ht_a_l = get_last_5_ht_stats(df, away)

        col_ht1, col_ht2 = st.columns(2)
        with col_ht1:
            st.write(f"**{home}**")
            st.write(f"Galibiyet: {ht_h_w} | Beraberlik: {ht_h_d} | Mağlubiyet: {ht_h_l}")
        with col_ht2:
            st.write(f"**{away}**")
            st.write(f"Galibiyet: {ht_a_w} | Beraberlik: {ht_a_d} | Mağlubiyet: {ht_a_l}")

        st.markdown("---")

        if st.button("Analiz Et ve Tahminle", type="primary"):
            if home == away:
                st.error("Farklı takımlar seçiniz.")
            else:
                res = predictor.predict_match(home, away)
                if res:
                    # 0. POISSON MATEMATİĞİ (YENİ)
                    if res['poisson']:
                        ph, pa = res['poisson']
                        st.info(f"**Poisson Matematiği:** Bu maçta **{home}** takımının **{ph:.2f}** gol, "
                                f"**{away}** takımının **{pa:.2f}** gol atması bekleniyor.")
                    
                    # 1. SKOR
                    st.subheader("Skor Tahmini")
                    s1, s2, s3 = st.columns(3)
                    top = res['top_scores']
                    if len(top) >= 3:
                        with s1: st.markdown(f"<div class='prediction-card'>1. <br><span class='score-badge'>{top[0]['score']}</span><br>%{top[0]['prob']:.1f}</div>", unsafe_allow_html=True)
                        with s2: st.markdown(f"<div class='prediction-card'>2. <br><span class='score-badge'>{top[1]['score']}</span><br>%{top[1]['prob']:.1f}</div>", unsafe_allow_html=True)
                        with s3: st.markdown(f"<div class='prediction-card'>3. <br><span class='score-badge'>{top[2]['score']}</span><br>%{top[2]['prob']:.1f}</div>", unsafe_allow_html=True)

                    # 2. MS
                    st.subheader("Maç Sonucu (MS)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Ev Sahibi", f"%{res['prob_home']:.1f}")
                    c2.metric("Beraberlik", f"%{res['prob_draw']:.1f}")
                    c3.metric("Deplasman", f"%{res['prob_away']:.1f}")
                    st.progress(int(res['prob_home']))
                    
                    # GÜVEN SKORU
                    conf = res.get('confidence', 0)
                    if conf >= 60:
                        st.success(f"🛡️ **Model Güven Skoru:** %{conf:.1f} (Yüksek Güven)")
                    elif conf >= 45:
                        st.warning(f"🛡️ **Model Güven Skoru:** %{conf:.1f} (Orta Güven)")
                    else:
                        st.error(f"🛡️ **Model Güven Skoru:** %{conf:.1f} (Düşük Güven - Riskli)")
                    
                    # 3. IY
                    st.subheader("İlk Yarı (IY)")
                    h1, h2, h3 = st.columns(3)
                    h1.metric("IY 1", f"%{res['prob_ht_home']:.1f}")
                    h2.metric("IY 0", f"%{res['prob_ht_draw']:.1f}")
                    h3.metric("IY 2", f"%{res['prob_ht_away']:.1f}")
                    
                    # 4. KARTLAR
                    st.markdown("<br>", unsafe_allow_html=True)
                    k1, k2 = st.columns(2)
                    with k1: st.markdown(f"""<div class="prediction-card"><h4>Toplam Gol 2.5</h4><h3 style="color:#10b981">ÜST: %{res['prob_over_2_5']:.1f}</h3><h5 style="color:#ef4444">ALT: %{res['prob_under_2_5']:.1f}</h5></div>""", unsafe_allow_html=True)
                    with k2: st.markdown(f"""<div class="prediction-card"><h4>Karşılıklı Gol</h4><h3 style="color:#10b981">VAR: %{res['prob_btts_yes']:.1f}</h3><h5 style="color:#ef4444">YOK: %{res['prob_btts_no']:.1f}</h5></div>""", unsafe_allow_html=True)
                    
                    # EXCEL
                    st.markdown("---")
                    excel_data = create_excel_download(res, home, away)
                    st.download_button(label="Analiz Raporunu İndir (Excel)", data=excel_data, file_name=f"{home}_vs_{away}_Analiz.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("Yetersiz veri.")
