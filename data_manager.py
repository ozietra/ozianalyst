import requests
import sqlite3
import time
import os
import pandas as pd
from datetime import datetime

# --- AYARLAR ---
DEFAULT_API_KEY = '7aa71c6f3c234e5da10884ffd6584eed'  # Varsayılan API Token
BASE_URL = 'https://api.football-data.org/v4'
DB_NAME = 'football_data.db'

# Takip Edilecek Ligler
LEAGUES = ['CL', 'BL1', 'DED', 'EC', 'SA', 'PL', 'FL1']

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Tablo yapısı: Maç sonuçları, İlk Yarı skorları ve Statü
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY,
            competition TEXT,
            season INTEGER,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            ht_home_score INTEGER,
            ht_away_score INTEGER,
            status TEXT
        )
    ''')
    
    # Gol Krallığı Tablosu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scorers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            competition TEXT,
            player_name TEXT,
            team_name TEXT,
            goals INTEGER,
            assists INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def fetch_and_store_data(api_key=None, seasons=[2023, 2024, 2025, 2026]):
    """API'den geçmiş ve gelecek maçları çeker."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    current_api_key = api_key if api_key else DEFAULT_API_KEY
    headers = {'X-Auth-Token': current_api_key}
    
    print("--- Veri Güncelleme Başladı ---")
    
    # Önce eski gol krallığı verilerini temizle (Güncel kalması için)
    cursor.execute('DELETE FROM scorers')
    conn.commit()
    
    for league in LEAGUES:
        # --- GOL KRALLIĞI VERİSİNİ ÇEK ---
        print(f"Gol Krallığı taranıyor: {league}")
        try:
            s_url = f"{BASE_URL}/competitions/{league}/scorers"
            s_res = requests.get(s_url, headers=headers)
            if s_res.status_code == 200:
                scorers = s_res.json().get('scorers', [])
                for s in scorers:
                    p_name = s['player']['name']
                    t_name = s['team']['name']
                    goals = s['goals']
                    assists = s.get('assists') or 0 # Asist verisi yoksa 0 yap
                    
                    cursor.execute('INSERT INTO scorers (competition, player_name, team_name, goals, assists) VALUES (?, ?, ?, ?, ?)', 
                                   (league, p_name, t_name, goals, assists))
                conn.commit()
            elif s_res.status_code == 429:
                time.sleep(60)
        except Exception as e:
            print(f"Gol Krallığı Hatası ({league}): {e}")
        
        time.sleep(6) # API Limit Koruması

        for season in seasons:
            print(f"Taranıyor: {league} - Sezon: {season}")
            url = f"{BASE_URL}/competitions/{league}/matches?season={season}"
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get('matches', [])
                    
                    for match in matches:
                        # Hem bitmiş hem de planlanmış maçları kaydet
                        status = match['status']
                        if status in ['FINISHED', 'SCHEDULED', 'TIMED', 'IN_PLAY', 'PAUSED']:
                            match_id = match['id']
                            match_date = match['utcDate']
                            home_team = match['homeTeam']['name']
                            away_team = match['awayTeam']['name']
                            
                            # Skorlar (Gelecek maçlar için None olabilir)
                            ft_home = match['score']['fullTime']['home']
                            ft_away = match['score']['fullTime']['away']
                            ht_home = match['score']['halfTime']['home']
                            ht_away = match['score']['halfTime']['away']
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO matches 
                                (id, competition, season, date, home_team, away_team, 
                                 home_score, away_score, ht_home_score, ht_away_score, status)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (match_id, league, season, match_date, home_team, away_team, 
                                  ft_home, ft_away, ht_home, ht_away, status))
                    conn.commit()
                elif response.status_code == 429:
                    print("!!! Hız Sınırı (Rate Limit). 60 saniye bekleniyor...")
                    time.sleep(60)
            except Exception as e:
                print(f"Hata: {e}")

            # API nezaket kuralı (Dakikada 10 istek sınırı için bekleme)
            time.sleep(7) 
            
    conn.close()
    print("--- Veri Güncelleme Tamamlandı ---")

def get_matches_df():
    """Model eğitimi için sadece BİTMİŞ maçları getirir."""
    if not os.path.exists(DB_NAME):
        init_db()
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM matches WHERE status='FINISHED' ORDER BY date DESC", conn)
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_next_fixtures(league_code, limit=10):
    """Gelecek maçları (Fikstür) getirir."""
    conn = get_db_connection()
    try:
        now = datetime.utcnow().isoformat()
        # Bugünden sonraki, bitmemiş maçları tarih sırasına göre çek
        query = """
            SELECT date, home_team, away_team 
            FROM matches 
            WHERE competition = ? AND status != 'FINISHED' AND date >= ?
            ORDER BY date ASC LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(league_code, now, limit))
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_live_matches(api_key=None):
    """Canlı oynanan maçları çeker."""
    current_api_key = api_key if api_key else DEFAULT_API_KEY
    headers = {'X-Auth-Token': current_api_key}
    
    # Sadece takip ettiğimiz liglerdeki canlı maçları filtrelemek için
    url = f"{BASE_URL}/matches?status=IN_PLAY,PAUSED"
    
    live_matches = []
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            
            for match in matches:
                comp_code = match['competition'].get('code')
                if comp_code in LEAGUES:
                    home = match['homeTeam']['name']
                    away = match['awayTeam']['name']
                    
                    # Skor kontrolü (None gelebilir)
                    score_home = match['score']['fullTime']['home']
                    score_away = match['score']['fullTime']['away']
                    if score_home is None: score_home = 0
                    if score_away is None: score_away = 0
                    
                    live_matches.append({
                        'lig': comp_code,
                        'ev': home,
                        'deplasman': away,
                        'skor': f"{score_home} - {score_away}"
                    })
    except Exception as e:
        print(f"Canlı maç hatası: {e}")
        
    return live_matches

def get_top_scorers(league_code):
    """Seçilen ligin gol krallığını getirir."""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("SELECT player_name as Oyuncu, team_name as Takım, goals as Gol, assists as Asist FROM scorers WHERE competition = ? ORDER BY goals DESC LIMIT 15", conn, params=(league_code,))
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def calculate_standings(df, league_code):
    """Veritabanındaki maçlardan Puan Durumu hesaplar."""
    if df.empty: return pd.DataFrame()
    
    league_df = df[df['competition'] == league_code]
    if league_df.empty: return pd.DataFrame()
    
    # En son sezonu bul
    latest_season = league_df['season'].max()
    matches = league_df[(league_df['season'] == latest_season) & (league_df['status'] == 'FINISHED')]
    
    teams = {}
    for _, match in matches.iterrows():
        home, away = match['home_team'], match['away_team']
        h_score, a_score = match['home_score'], match['away_score']
        
        if home not in teams: teams[home] = {'O':0,'G':0,'B':0,'M':0,'A':0,'Y':0,'P':0}
        if away not in teams: teams[away] = {'O':0,'G':0,'B':0,'M':0,'A':0,'Y':0,'P':0}
        
        teams[home]['O']+=1; teams[away]['O']+=1
        teams[home]['A']+=h_score; teams[home]['Y']+=a_score
        teams[away]['A']+=a_score; teams[away]['Y']+=h_score
        
        if h_score > a_score:
            teams[home]['G']+=1; teams[home]['P']+=3; teams[away]['M']+=1
        elif a_score > h_score:
            teams[away]['G']+=1; teams[away]['P']+=3; teams[home]['M']+=1
        else:
            teams[home]['B']+=1; teams[home]['P']+=1; teams[away]['B']+=1; teams[away]['P']+=1
            
    standings_df = pd.DataFrame.from_dict(teams, orient='index')
    if not standings_df.empty:
        standings_df['Av'] = standings_df['A'] - standings_df['Y']
        # Sıralama: Puan > Averaj > Atılan Gol
        standings_df = standings_df.sort_values(by=['P', 'Av', 'A'], ascending=False)
        standings_df.reset_index(inplace=True)
        standings_df.rename(columns={'index': 'Takım'}, inplace=True)
        standings_df.index += 1 
    return standings_df