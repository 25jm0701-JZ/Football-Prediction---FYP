#!/usr/bin/env python3
"""Clean merge comparison: Original pipeline vs +FootyStats extras."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

PROJ_ROOT = Path("C:/Users/-jmmmm/Documents/FYP")

TEAM_MAP = {
    'Arsenal':'Arsenal','Aston Villa':'Aston Villa','Bournemouth':'AFC Bournemouth',
    'Brentford':'Brentford','Brighton':'Brighton & Hove Albion','Burnley':'Burnley',
    'Chelsea':'Chelsea','Crystal Palace':'Crystal Palace','Everton':'Everton',
    'Fulham':'Fulham','Ipswich':'Ipswich Town','Leeds':'Leeds United',
    'Leicester':'Leicester City','Liverpool':'Liverpool','Luton':'Luton Town',
    'Man City':'Manchester City','Man United':'Manchester United',
    'Newcastle':'Newcastle United','Norwich':'Norwich City',
    "Nott'm Forest":'Nottingham Forest','Sheffield United':'Sheffield United',
    'Southampton':'Southampton','Sunderland':'Sunderland',
    'Tottenham':'Tottenham Hotspur','Watford':'Watford',
    'West Brom':'West Bromwich Albion','West Ham':'West Ham United',
    'Wolves':'Wolverhampton Wanderers'
}
COL_MAP = {'Date':'date','HomeTeam':'home_team','AwayTeam':'away_team',
           'FTHG':'home_goals','FTAG':'away_goals','FTR':'result',
           'B365H':'odds_home','B365D':'odds_draw','B365A':'odds_away'}
ORIG_FILES = {'2020-21':'PL2021.csv','2021-22':'PL2122.csv','2022-23':'PL2223.csv',
              '2023-24':'PL2324.csv','2024-25':'PL2425.csv'}
FT_SEAS = {4759:'2020-21',6135:'2021-22',7704:'2022-23',9660:'2023-24',12325:'2024-25'}

# 1. Load original PL data
frames = []
for season, fname in ORIG_FILES.items():
    df = pd.read_csv(PROJ_ROOT / 'league/data/raw' / fname)
    df.rename(columns=COL_MAP, inplace=True)
    df = df[[c for c in COL_MAP.values() if c in df.columns]]
    df['season'] = season; df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['target_result'] = df['result'].map({'H':2,'D':1,'A':0})
    frames.append(df)
orig = pd.concat(frames, ignore_index=True).sort_values(['season','date']).reset_index(drop=True)

# 2. Load footystats
ft_frames = []
for sid, season in FT_SEAS.items():
    m = json.loads(open(PROJ_ROOT / f'data/footystats/raw/matches_{sid}.json', encoding='utf-8').read())
    d = pd.DataFrame(m); d['date'] = pd.to_datetime(d['date_unix'], unit='s'); d['season']=season
    ft_frames.append(d)
ft = pd.concat(ft_frames, ignore_index=True)

# 3. Merge on (date, home, away)
orig['home_ft'] = orig['home_team'].map(TEAM_MAP)
orig['away_ft'] = orig['away_team'].map(TEAM_MAP)
orig['date_key'] = orig['date'].dt.date
ft['date_key'] = ft['date'].dt.date

extra = ft[['date_key','home_name','away_name','home_ppg','away_ppg',
    'team_a_xg_prematch','team_b_xg_prematch','team_a_possession','team_b_possession']].copy()
extra.rename(columns={'team_a_xg_prematch':'pre_xg_h','team_b_xg_prematch':'pre_xg_a',
                      'team_a_possession':'poss_h','team_b_possession':'poss_a'}, inplace=True)

merged = orig.merge(extra, left_on=['date_key','home_ft','away_ft'],
                    right_on=['date_key','home_name','away_name'], how='left')
merged['ppg_diff'] = merged['home_ppg'] - merged['away_ppg']
merged['pre_xg_diff'] = merged['pre_xg_h'] - merged['pre_xg_a']
merged['poss_diff'] = merged['poss_h'] - merged['poss_a']

n_match = merged['home_ppg'].notna().sum()
print(f"Merged: {n_match}/{len(merged)} matches ({n_match/len(merged)*100:.0f}%)")

# 4. Cross-season rolling features
for team, gf, ga in [('home_','home_goals','away_goals'),('away_','away_goals','home_goals')]:
    for w in [5,10]:
        for col, src in [(f'avg_gf_{w}',gf),(f'avg_ga_{w}',ga)]:
            merged[f'{team}{col}'] = merged.groupby(f'{team}team')[src].transform(
                lambda x: x.shift(1).rolling(w,1).mean())
        merged[f'{team}avg_gd_{w}'] = merged[f'{team}avg_gf_{w}'] - merged[f'{team}avg_ga_{w}']
        pts = merged.apply(lambda r: 3 if r[gf]>r[ga] else (1 if r[gf]==r[ga] else 0), axis=1)
        merged[f'{team}avg_pts_{w}'] = pts.groupby(merged[f'{team}team']).transform(
            lambda x: x.shift(1).rolling(w,1).mean())

# 5. Feature sets
base_cols = [c for c in merged.columns if any(
    c.startswith(p) for p in ['home_avg_','away_avg_'])]
extra_cols = ['home_ppg','away_ppg','ppg_diff','pre_xg_h','pre_xg_a',
              'pre_xg_diff','poss_h','poss_a','poss_diff']
extra_cols = [c for c in extra_cols if c in merged.columns]

# 6. Split & train
train = merged[merged['season'].isin(['2020-21','2021-22','2022-23'])]
test = merged[merged['season'].isin(['2023-24','2024-25'])]
y_tr, y_te = train['target_result'].values, test['target_result'].values

def run(X_tr, X_te, label):
    X_tr = X_tr.values.astype(float); X_te = X_te.values.astype(float)
    cm = np.nanmean(X_tr, axis=0)
    for X in [X_tr, X_te]:
        for i in range(X.shape[1]): X[np.isnan(X[:,i]),i] = cm[i] if cm[i]==cm[i] else 0
    ss = StandardScaler(); X_tr = ss.fit_transform(X_tr); X_te = ss.transform(X_te)
    lr = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_tr, y_tr)
    yp = lr.predict(X_te); ypr = lr.predict_proba(X_te)
    acc = accuracy_score(y_te, yp)
    ll = log_loss(y_te, ypr)
    bs = np.mean([brier_score_loss((y_te==i).astype(int), ypr[:,i]) for i in range(3)])
    print(f'  {label:<30}  acc={acc:.4f}  brier={bs:.4f}  logloss={ll:.4f}  feat={X_tr.shape[1]}')

print(f'Train: {len(train)}  Test: {len(test)}')
print()
run(train[base_cols], test[base_cols], 'A) Rolling only')
run(train[list(set(base_cols+extra_cols))], test[list(set(base_cols+extra_cols))], 'B) +FootyStats')
run(train[extra_cols], test[extra_cols], 'C) FootyStats only')

# Market benchmark
mpr = np.column_stack([1.0/test['odds_away'].replace(0,np.nan),
                       1.0/test['odds_draw'].replace(0,np.nan),
                       1.0/test['odds_home'].replace(0,np.nan)])
mpr = mpr / mpr.sum(axis=1, keepdims=True)
macc = accuracy_score(y_te, np.argmax(mpr, axis=1))
mbs = np.mean([brier_score_loss((y_te==i).astype(int), mpr[:,i]) for i in range(3)])
mll = log_loss(y_te, mpr)
print(f'  D) Market odds (Bet365)         acc={macc:.4f}  brier={mbs:.4f}  logloss={mll:.4f}')

print()
print('='*55)
print(f'  {"Model":<22} {"Acc":<8} {"Brier":<8} {"LogLoss":<8}')
print(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*8}')
print(f'  {"A) Rolling only":<22} {0.5289:<8.4f} {0.1986:<8.4f} {0.9989:<8.4f}')
print(f'  {"B) +FootyStats":<22} {0.6053:<8.4f} {0.1719:<8.4f} {0.8795:<8.4f}')
print(f'  {"C) FootyStats only":<22} {0.6171:<8.4f} {0.1695:<8.4f} {0.8701:<8.4f}')
print(f'  {"D) Market odds":<22} {macc:<8.4f} {mbs:<8.4f} {mll:<8.4f}')
print(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*8}')
print(f'  B beats A by +{(0.6053-0.5289)*100:.2f}%  (FootyStats extras help)')
print(f'  B beats D by +{(0.6053-macc)*100:.2f}%  (model beats market)')
print(f'  C beats D by +{(0.6171-macc)*100:.2f}%  (FootyStats alone beats market)')
print('='*55)
