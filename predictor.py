import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import poisson

class MatchPredictor:
    def __init__(self, df):
        self.df = df.copy()
        # Modeller
        self.model_result = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model_over_under = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model_btts = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model_ht_result = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
        # Skor Tahmini için ayrı modeller
        self.model_home_goals = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.model_away_goals = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        
        self.le_team = LabelEncoder()
        self.is_trained = False
        # Yeni özellikler eklendi: Form ve Gol Ortalamaları
        self.features = ['home_code', 'away_code', 'season_code', 'home_form', 'away_form', 'home_gs_avg', 'away_gs_avg']
        self.accuracy = 0.0
        
    def prepare_features(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Hedefler
        self.df['result'] = np.where(self.df['home_score'] > self.df['away_score'], 2,
                                     np.where(self.df['home_score'] == self.df['away_score'], 1, 0))
        
        self.df['ht_result'] = np.where(self.df['ht_home_score'] > self.df['ht_away_score'], 2,
                                     np.where(self.df['ht_home_score'] == self.df['ht_away_score'], 1, 0))
        
        self.df['total_goals'] = self.df['home_score'] + self.df['away_score']
        self.df['over_2_5'] = (self.df['total_goals'] > 2.5).astype(int)
        self.df['btts'] = ((self.df['home_score'] > 0) & (self.df['away_score'] > 0)).astype(int)
        
        # Skor sınıfları (4 ve üzeri golleri '4' olarak grupla)
        self.df['home_goals_class'] = self.df['home_score'].apply(lambda x: 4 if x >= 4 else x)
        self.df['away_goals_class'] = self.df['away_score'].apply(lambda x: 4 if x >= 4 else x)
        
        # Encoding
        all_teams = pd.concat([self.df['home_team'], self.df['away_team']]).unique()
        self.le_team.fit(all_teams)
        
        self.df['home_code'] = self.le_team.transform(self.df['home_team'])
        self.df['away_code'] = self.le_team.transform(self.df['away_team'])
        self.df['season_code'] = self.df['season'] - 2020 
        
        # --- FORM VE İSTATİSTİK HESAPLAMA (YENİ) ---
        self.add_form_features()
        
    def add_form_features(self):
        """Takımların son 5 maçlık form ve gol istatistiklerini hesaplar."""
        # Veriyi uzun formata çevir (Her maç için 2 satır: Ev ve Deplasman takımı için ayrı ayrı)
        home_stats = self.df[['date', 'home_team', 'home_score', 'away_score']].rename(
            columns={'home_team': 'team', 'home_score': 'goals_for', 'away_score': 'goals_against'})
        home_stats['points'] = np.where(home_stats['goals_for'] > home_stats['goals_against'], 3, 
                                        np.where(home_stats['goals_for'] == home_stats['goals_against'], 1, 0))
        
        away_stats = self.df[['date', 'away_team', 'away_score', 'home_score']].rename(
            columns={'away_team': 'team', 'away_score': 'goals_for', 'home_score': 'goals_against'})
        away_stats['points'] = np.where(away_stats['goals_for'] > away_stats['goals_against'], 3, 
                                        np.where(away_stats['goals_for'] == away_stats['goals_against'], 1, 0))
        
        # Tüm maçları birleştir ve tarihe göre sırala
        stats_df = pd.concat([home_stats, away_stats]).sort_values(['team', 'date'])
        
        # Rolling (Kayar Pencere) Hesaplama
        # shift(1) kullanıyoruz çünkü bugünkü maçı tahmin ederken bugünkü sonucu bilemeyiz, dünü bilmeliyiz.
        stats_df['form_5'] = stats_df.groupby('team')['points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)
        stats_df['gs_avg_5'] = stats_df.groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)
        
        # En son güncel form durumlarını tahminlerde kullanmak için sakla
        # (Son oynanan maçtan sonraki durum)
        self.latest_team_stats = stats_df.groupby('team').last()[['form_5', 'gs_avg_5']].to_dict('index')
        
        # Hesaplanan verileri ana tabloya (self.df) geri birleştir
        stats_df_merge = stats_df[['date', 'team', 'form_5', 'gs_avg_5']]
        
        self.df = self.df.merge(stats_df_merge.rename(columns={'team': 'home_team', 'form_5': 'home_form', 'gs_avg_5': 'home_gs_avg'}), 
                                on=['date', 'home_team'], how='left')
        
        self.df = self.df.merge(stats_df_merge.rename(columns={'team': 'away_team', 'form_5': 'away_form', 'gs_avg_5': 'away_gs_avg'}), 
                                on=['date', 'away_team'], how='left')
        
        # İlk maçlarda veri olmadığı için NaN oluşabilir, 0 ile doldur
        self.df = self.df.fillna(0)
        
    def train(self):
        if len(self.df) < 10: return False
        self.prepare_features()
        
        X = self.df[self.features]
        y = self.df['result']
        
        # Doğruluk oranı hesaplamak için veriyi böl (Test seti %20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Eğitim
        self.model_result.fit(X_train, y_train)
        self.model_over_under.fit(X, self.df['over_2_5'])
        self.model_btts.fit(X, self.df['btts'])
        self.model_ht_result.fit(X, self.df['ht_result'])
        self.model_home_goals.fit(X, self.df['home_goals_class'])
        self.model_away_goals.fit(X, self.df['away_goals_class'])
        
        # Başarı Oranını Hesapla
        self.accuracy = self.model_result.score(X_test, y_test) * 100
        
        # Tüm veriyle tekrar eğit (Maksimum performans için)
        self.model_result.fit(X, y)
        
        self.is_trained = True
        return True
        
    def predict_poisson(self, home_team, away_team, league_code):
        """Poisson dağılımı ile matematiksel skor tahmini."""
        # Lig ortalamalarını bul
        league_matches = self.df[self.df['competition'] == league_code]
        if league_matches.empty: return None
        
        avg_home_goals = league_matches['home_score'].mean()
        avg_away_goals = league_matches['away_score'].mean()
        
        # Takım ortalamaları
        home_m = league_matches[league_matches['home_team'] == home_team]
        away_m = league_matches[league_matches['away_team'] == away_team]
        
        if home_m.empty or away_m.empty: return None
        
        # Hücum ve Savunma Güçleri
        h_att = home_m['home_score'].mean() / avg_home_goals
        h_def = home_m['away_score'].mean() / avg_away_goals
        
        a_att = away_m['away_score'].mean() / avg_away_goals
        a_def = away_m['home_score'].mean() / avg_home_goals
        
        # Beklenen Gol Sayıları (Lambda)
        home_expect = h_att * a_def * avg_home_goals
        away_expect = a_att * h_def * avg_away_goals
        
        return home_expect, away_expect

    def predict_match(self, home_team, away_team, current_season=2026):
        if not self.is_trained: return None
        try:
            h_code = self.le_team.transform([home_team])[0]
            a_code = self.le_team.transform([away_team])[0]
            s_code = current_season - 2020
            
            # Lig kodunu bul (Poisson için)
            league_code = self.df[self.df['home_team'] == home_team]['competition'].iloc[0]
            
            # Takımların son form durumlarını çek
            h_stats = self.latest_team_stats.get(home_team, {'form_5': 0, 'gs_avg_5': 0})
            a_stats = self.latest_team_stats.get(away_team, {'form_5': 0, 'gs_avg_5': 0})
            
            # Girdi verisini hazırla (Yeni özelliklerle)
            input_data = pd.DataFrame([[h_code, a_code, s_code, h_stats['form_5'], a_stats['form_5'], h_stats['gs_avg_5'], a_stats['gs_avg_5']]], columns=self.features)
            
            # Olasılıklar
            probs_result = self.model_result.predict_proba(input_data)[0]
            probs_over = self.model_over_under.predict_proba(input_data)[0]
            probs_btts = self.model_btts.predict_proba(input_data)[0]
            probs_ht = self.model_ht_result.predict_proba(input_data)[0]
            
            # Güven Skoru (En yüksek olasılık)
            confidence_score = max(probs_result) * 100
            
            # Skor Tahmini (Kombinasyon Hesabı)
            h_goal_probs = self.model_home_goals.predict_proba(input_data)[0]
            a_goal_probs = self.model_away_goals.predict_proba(input_data)[0]
            
            score_probs = []
            for h_g, h_p in enumerate(h_goal_probs):
                for a_g, a_p in enumerate(a_goal_probs):
                    score_label = f"{h_g}-{a_g}"
                    if h_g == 4: score_label = "4+-"+str(a_g)
                    if a_g == 4: score_label = str(h_g)+"-4+"
                    score_probs.append({'score': score_label, 'prob': h_p * a_p * 100})
            
            score_probs.sort(key=lambda x: x['prob'], reverse=True)
            
            # Poisson Tahmini
            poisson_stats = self.predict_poisson(home_team, away_team, league_code)
            
            return {
                'prob_home': probs_result[2] * 100,
                'prob_draw': probs_result[1] * 100,
                'prob_away': probs_result[0] * 100,
                'prob_over_2_5': probs_over[1] * 100,
                'prob_under_2_5': probs_over[0] * 100,
                'prob_btts_yes': probs_btts[1] * 100,
                'prob_btts_no': probs_btts[0] * 100,
                'prob_ht_home': probs_ht[2] * 100,
                'prob_ht_draw': probs_ht[1] * 100,
                'prob_ht_away': probs_ht[0] * 100,
                'top_scores': score_probs[:3],
                'poisson': poisson_stats,
                'model_accuracy': self.accuracy,
                'confidence': confidence_score
            }
        except Exception as e:
            print(f"Tahmin Hatası: {e}")
            return None