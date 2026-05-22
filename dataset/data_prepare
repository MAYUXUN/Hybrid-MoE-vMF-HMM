# -*- coding: utf-8 -*-
"""
GETNext-style data preparation for Foursquare NYC dataset (TSMC2014).
Reference: Yang et al., "Where to Go Next: Modeling Long- and Short-Term User
           Preferences for Point-of-Interest Recommendation", SIGIR 2022, §5.1.1.

Input format (dataset_TSMC2014_NYC.txt, tab-separated, no header):
  uid  venue_id  category_id  category  lat  lon  tz_offset_min  created_at_utc

Outputs
-------
  NYC_data.pkl   — PKL consumed by MF / FPMC / HMHMM / GETNext
  graph_X.csv             — POI node features for GETNext graph construction
  NYC_train.csv           — check-in level CSV, compatible with GETNext train.py
  NYC_val.csv
  NYC_test.csv
"""

import os
import random
import pickle
import numpy as np
import pandas as pd


# =========================================================
# CONFIG  —  all hyper-parameters in one place
# =========================================================
CONFIG = {
    'input_txt':       '.',
    'output_dir':      '.',
    'output_pkl':      '.',
    'poi_min_checkin': 10,
    'user_min_checkin':10,
    'traj_gap_hours':  24,
    'traj_min_len':    3,
    'train_ratio':     0.8,
    'val_ratio':       0.1,
    'test_ratio':      0.1,
    'dataset':         'NYC',  # used for output file naming
    'seed':            42,
}


if __name__ == '__main__':

    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # =========================================================
    # 1. Load raw data
    # =========================================================
    print('=== Step 1: Loading raw data ===')

    # NYC dataset: no header, 8 tab-separated columns
    col_names = ['uid', 'venue_id', 'category_id', 'category',
                 'lat', 'lon', 'tz_offset_min', 'created_at']
    df = pd.read_csv(CONFIG['input_txt'], sep='\t', header=None,
                     names=col_names, encoding='latin-1')

    # Ensure uid is stored as string so trajectory_id.split('_')[0] works
    df['uid'] = df['uid'].astype(str)

    # Parse UTC timestamp ("Tue Apr 03 18:00:09 +0000 2012") → local time
    df['created_at'] = pd.to_datetime(
        df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y', utc=True)
    df['tz_offset_min'] = pd.to_numeric(df['tz_offset_min'], errors='coerce')
    # local_dt = UTC + tz_offset_min  (tz_offset_min is negative for NYC, e.g. -240)
    df['local_dt'] = (df['created_at']
                      + pd.to_timedelta(df['tz_offset_min'], unit='m')
                      ).dt.tz_localize(None)

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['venue_id']    = df['venue_id'].astype(str)
    df['category_id'] = df['category_id'].astype(str)
    df = df.dropna(subset=['uid', 'venue_id', 'category_id', 'category',
                           'lat', 'lon', 'local_dt', 'tz_offset_min']).copy()
    df = df.sort_values(['uid', 'local_dt']).reset_index(drop=True)

    # Unix timestamp (seconds)
    df['unix_ts'] = (df['local_dt'].astype(np.int64) // 10 ** 9).astype(int)

    print(f'  Raw: {df["uid"].nunique()} users, '
          f'{df["venue_id"].nunique()} POIs, {len(df)} check-ins')

    # =========================================================
    # 2. Filter: POI global count >= poi_min_checkin  (§5.1.1 step 1)
    # =========================================================
    print('=== Step 2: POI filter ===')
    poi_counts = df['venue_id'].value_counts()
    valid_pois = poi_counts[poi_counts >= CONFIG['poi_min_checkin']].index
    df = df[df['venue_id'].isin(valid_pois)].copy()
    print(f'  After POI filter: {df["venue_id"].nunique()} POIs, {len(df)} check-ins')

    # =========================================================
    # 3. Filter: user remaining count >= user_min_checkin  (§5.1.1 step 2)
    # =========================================================
    print('=== Step 3: User filter ===')
    user_counts = df['uid'].value_counts()
    valid_users = user_counts[user_counts >= CONFIG['user_min_checkin']].index
    df = df[df['uid'].isin(valid_users)].copy()
    print(f'  After user filter: {df["uid"].nunique()} users, {len(df)} check-ins')

    n_users_filtered = df['uid'].nunique()
    n_pois_filtered  = df['venue_id'].nunique()
    n_ci_filtered    = len(df)

    # =========================================================
    # 4. Trajectory segmentation: split on gap > 24 h  (§5.1.1)
    #    trajectory_id = f'{uid}_{seq_no}'
    #    Important: traj_id.split('_')[0] must equal uid (string, no underscores
    #    assumed in raw uid — Foursquare uids are numeric)
    # =========================================================
    print('=== Step 4: Trajectory segmentation (24-h gap) ===')
    gap_threshold_sec = CONFIG['traj_gap_hours'] * 3600

    raw_trajectories = []

    for uid, group in df.groupby('uid', sort=False):
        group = group.sort_values('local_dt').reset_index(drop=True)
        seq_no = 0
        current_rows = [group.iloc[0]]

        for i in range(1, len(group)):
            gap = (group.iloc[i]['local_dt'] - group.iloc[i - 1]['local_dt']).total_seconds()
            if gap > gap_threshold_sec:
                raw_trajectories.append({
                    'trajectory_id': f'{uid}_{seq_no}',
                    'uid_raw':       uid,           # always str
                    'rows':          current_rows,
                    'first_dt':      current_rows[0]['local_dt'],
                })
                seq_no += 1
                current_rows = [group.iloc[i]]
            else:
                current_rows.append(group.iloc[i])

        raw_trajectories.append({
            'trajectory_id': f'{uid}_{seq_no}',
            'uid_raw':       uid,
            'rows':          current_rows,
            'first_dt':      current_rows[0]['local_dt'],
        })

    print(f'  Before length filter: {len(raw_trajectories)} trajectories')

    # =========================================================
    # 5. Filter trajectories with length < traj_min_len
    # =========================================================
    raw_trajectories = [t for t in raw_trajectories
                        if len(t['rows']) >= CONFIG['traj_min_len']]
    print(f'  After length filter (>={CONFIG["traj_min_len"]}): '
          f'{len(raw_trajectories)} trajectories')

    # =========================================================
    # 6. Global 80/10/10 split by first check-in time  (§5.1.1)
    #    Assign each trajectory as a whole to one split; cut by
    #    cumulative check-in count so ratio is by check-ins not trajs.
    # =========================================================
    print('=== Step 5: Train / Val / Test split (80/10/10 by time) ===')
    raw_trajectories.sort(key=lambda x: x['first_dt'])

    total_ci  = sum(len(t['rows']) for t in raw_trajectories)
    train_cut = int(total_ci * CONFIG['train_ratio'])
    val_cut   = int(total_ci * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

    train_raw, val_raw, test_raw = [], [], []
    cumulative = 0
    for traj in raw_trajectories:
        if cumulative < train_cut:
            train_raw.append(traj)
        elif cumulative < val_cut:
            val_raw.append(traj)
        else:
            test_raw.append(traj)
        cumulative += len(traj['rows'])

    def traj_time_range(split):
        all_dts = [r['local_dt'] for t in split for r in t['rows']]
        return min(all_dts), max(all_dts)

    for name, split in [('Train', train_raw), ('Val  ', val_raw), ('Test ', test_raw)]:
        t0, t1 = traj_time_range(split)
        print(f'  {name}: {len(split):5d} traj, '
              f'{sum(len(t["rows"]) for t in split):7d} check-ins  '
              f'[{t0.date()} → {t1.date()}]')

    # =========================================================
    # 7. Build encodings from TRAIN set only  (§5.1.1 OOV handling)
    #    pid_dict : raw venue_id (str) -> int, 1-indexed (0 = padding)
    #    uid_dict : raw uid (str)      -> int, 0-indexed
    #    cid_dict : raw cat_id (str)   -> int, 0-indexed
    # =========================================================
    print('=== Step 6: Building encodings from train set only ===')

    train_pois, train_users, train_cats = set(), set(), set()
    poi_to_rawcat  = {}   # raw poi -> raw cat id
    rawcat_to_name = {}   # raw cat id -> category name
    poi_to_latlon  = {}   # raw poi -> (lat, lon)
    poi_train_cnt  = {}   # raw poi -> checkin count in train

    for traj in train_raw:
        for rec in traj['rows']:
            p, u, c = rec['venue_id'], rec['uid'], rec['category_id']
            train_pois.add(p);  train_users.add(u);  train_cats.add(c)
            poi_to_rawcat[p]   = c
            rawcat_to_name[c]  = rec['category']
            poi_to_latlon[p]   = (float(rec['lat']), float(rec['lon']))
            poi_train_cnt[p]   = poi_train_cnt.get(p, 0) + 1

    pid_dict = {p: idx + 1 for idx, p in enumerate(sorted(train_pois))}
    uid_dict = {u: idx     for idx, u in enumerate(sorted(train_users))}
    cid_dict = {c: idx     for idx, c in enumerate(sorted(train_cats))}

    pid_cid_dict = {pid_dict[p]: cid_dict[c]
                    for p, c in poi_to_rawcat.items()
                    if p in pid_dict and c in cid_dict}
    cid_cat_dict = {cid_dict[c]: name
                    for c, name in rawcat_to_name.items()
                    if c in cid_dict}
    pid_lat_lon  = {pid_dict[p]: ll
                    for p, ll in poi_to_latlon.items()
                    if p in pid_dict}

    print(f'  Encoded: {len(pid_dict)} POIs, '
          f'{len(uid_dict)} users, {len(cid_dict)} categories')
    print(f'  [Compare GETNext Table 1 NYC: ~1083 users, ~38333 POIs, ~251 cats]')

    # =========================================================
    # 8. Convert raw trajectories -> PKL sample dicts
    #    OOV user  → user_id = -1  (was: raw string; now always int or -1)
    #    OOV POI   → poi_enc = 0   (padding id)
    # =========================================================
    print('=== Step 7: Converting to GETNext-style samples ===')

    def convert(traj):
        uid_raw = traj['uid_raw']
        uid_enc = uid_dict.get(uid_raw, -1)      # -1 for OOV user

        poi_seq_raw  = []
        poi_seq      = []
        cat_seq      = []
        lat_seq      = []
        lon_seq      = []
        ts_seq       = []
        weekday_seq  = []
        hour_bin_seq = []
        norm_time_seq= []
        oov_mask     = []

        user_oov = uid_enc == -1

        for rec in traj['rows']:
            p  = rec['venue_id']
            c  = rec['category_id']
            dt = rec['local_dt']

            poi_enc = pid_dict.get(p, 0)
            cat_enc = cid_dict.get(c, 0)
            poi_oov = p not in pid_dict

            weekday  = dt.weekday() + 1                                         # 1~7
            hour_bin = int(dt.hour / 2) + 1                                     # 1~12
            norm_t   = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0

            poi_seq_raw.append(p)
            poi_seq.append(poi_enc)
            cat_seq.append(cat_enc)
            lat_seq.append(float(rec['lat']))
            lon_seq.append(float(rec['lon']))
            ts_seq.append(int(rec['unix_ts']))
            weekday_seq.append(weekday)
            hour_bin_seq.append(hour_bin)
            norm_time_seq.append(round(norm_t, 6))
            oov_mask.append(bool(user_oov or poi_oov))

        return {
            'trajectory_id':    traj['trajectory_id'],
            'user_id_raw':      uid_raw,           # original uid str
            'user_id':          uid_enc,            # int ≥ 0, or -1 for OOV
            'poi_seq_raw':      poi_seq_raw,        # original venue_id list
            'poi_seq':          poi_seq,
            'cat_seq':          cat_seq,
            'lat_seq':          lat_seq,
            'lon_seq':          lon_seq,
            'timestamp_seq':    ts_seq,
            'weekday_seq':      weekday_seq,
            'hour_bin_seq':     hour_bin_seq,
            'norm_in_day_time': norm_time_seq,
            'oov_mask':         oov_mask,
        }

    train_trajectories = [convert(t) for t in train_raw]
    val_trajectories   = [convert(t) for t in val_raw]
    test_trajectories  = [convert(t) for t in test_raw]

    # =========================================================
    # 9. Statistics
    # =========================================================
    n_train_ci = sum(len(s['poi_seq']) for s in train_trajectories)
    n_val_ci   = sum(len(s['poi_seq']) for s in val_trajectories)
    n_test_ci  = sum(len(s['poi_seq']) for s in test_trajectories)

    avg_train_len = n_train_ci / len(train_trajectories) if train_trajectories else 0

    def oov_stats(split_trajs):
        n_oov_user = sum(1 for s in split_trajs if s['user_id'] == -1)
        all_pois   = [p for s in split_trajs for p in s['poi_seq_raw']]
        n_oov_poi  = sum(1 for p in all_pois if p not in pid_dict)
        n_oov_traj = sum(1 for s in split_trajs if any(s['oov_mask']))
        return n_oov_user, n_oov_poi, len(all_pois), n_oov_traj

    val_oov_u,  val_oov_p,  val_total_p,  val_oov_traj  = oov_stats(val_trajectories)
    test_oov_u, test_oov_p, test_total_p, test_oov_traj = oov_stats(test_trajectories)

    stats = {
        'num_users':         len(uid_dict),
        'num_pois':          len(pid_dict),
        'num_cats':          len(cid_dict),
        'num_train_traj':    len(train_trajectories),
        'num_val_traj':      len(val_trajectories),
        'num_test_traj':     len(test_trajectories),
        'num_train_checkin': n_train_ci,
        'num_val_checkin':   n_val_ci,
        'num_test_checkin':  n_test_ci,
    }

    # =========================================================
    # 10. Save PKL
    # =========================================================
    dataset = {
        'train_trajectories': train_trajectories,
        'val_trajectories':   val_trajectories,
        'test_trajectories':  test_trajectories,
        'pid_dict':           pid_dict,
        'uid_dict':           uid_dict,
        'cid_dict':           cid_dict,
        'pid_cid_dict':       pid_cid_dict,
        'cid_cat_dict':       cid_cat_dict,
        'pid_lat_lon':        pid_lat_lon,
        'stats':              stats,
    }

    out_path = CONFIG['output_pkl']
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    pkl_mb = os.path.getsize(out_path) / 1024 / 1024

    # =========================================================
    # 11. Save graph_X.csv  (POI node features for GETNext graph)
    #     Only train POIs (graph is built from train transitions only)
    # =========================================================
    print('=== Step 8: Saving graph_X.csv ===')

    graph_rows = []
    for raw_poi, pid_enc in pid_dict.items():
        c_raw = poi_to_rawcat[raw_poi]
        lat, lon = poi_to_latlon[raw_poi]
        graph_rows.append({
            'node_name/poi_id': raw_poi,
            'checkin_cnt':      poi_train_cnt.get(raw_poi, 0),
            'poi_catid':        c_raw,
            'poi_catid_code':   cid_dict[c_raw],
            'latitude':         lat,
            'longitude':        lon,
        })

    graph_df = pd.DataFrame(graph_rows)
    graph_path = os.path.join(CONFIG['output_dir'], 'graph_X.csv')
    graph_df.to_csv(graph_path, index=False)
    print(f'  graph_X.csv: {len(graph_df)} POI nodes → {graph_path}')

    # =========================================================
    # 12. Save TKY_train/val/test.csv  (GETNext-compatible check-in CSVs)
    #     One row per check-in; all trajectories included (no OOV filter here).
    #     GETNext's TrajectoryDatasetVal handles OOV in-flight.
    # =========================================================
    print('=== Step 9: Saving split CSVs ===')

    def build_csv_rows(trajectories):
        rows = []
        for traj in trajectories:
            tid   = traj['trajectory_id']
            uid_r = traj['user_id_raw']
            for k in range(len(traj['poi_seq'])):
                rows.append({
                    'trajectory_id':  tid,
                    'user_id':        uid_r,
                    'poi_id':         traj['poi_seq_raw'][k],
                    'poi_catid':      '',   # filled below if needed
                    'poi_catid_code': traj['cat_seq'][k],
                    'latitude':       traj['lat_seq'][k],
                    'longitude':      traj['lon_seq'][k],
                    'UTC_time':       traj['timestamp_seq'][k],
                    'weekday':        traj['weekday_seq'][k],
                    'hour_bin':       traj['hour_bin_seq'][k],
                    'norm_in_day_time': traj['norm_in_day_time'][k],
                })
        return pd.DataFrame(rows)

    # Attach raw category name via pid_dict → cid_dict → cid_cat_dict
    raw_poi_to_cat_name = {p: rawcat_to_name[c] for p, c in poi_to_rawcat.items()}

    ds = CONFIG['dataset']
    csv_paths = {}
    for split_name, split_trajs in [('train', train_trajectories),
                                     ('val',   val_trajectories),
                                     ('test',  test_trajectories)]:
        cdf = build_csv_rows(split_trajs)
        cdf['poi_catid'] = cdf['poi_id'].map(poi_to_rawcat).fillna('')
        csv_p = os.path.join(CONFIG['output_dir'], f'{ds}_{split_name}.csv')
        cdf.to_csv(csv_p, index=False)
        csv_paths[split_name] = csv_p
        print(f'  {ds}_{split_name}.csv: {len(cdf):7d} rows → {csv_p}')

    # =========================================================
    # 13. Final summary
    # =========================================================
    print('\n==================== Summary ====================')
    print(f'Input  : {CONFIG["input_txt"]}')
    print(f'PKL    : {out_path}  ({pkl_mb:.2f} MB)')
    print(f'\nFilter results:')
    print(f'  {n_users_filtered} users, {n_pois_filtered} POIs, {n_ci_filtered} check-ins')
    print(f'\nStats (compare GETNext Table 1 NYC: ~1083 / ~38333 / ~251):')
    for k, v in stats.items():
        print(f'  {k}: {v}')
    print(f'\n  avg_train_traj_len : {avg_train_len:.2f}')
    print(f'\nOOV (val):')
    print(f'  OOV users      : {val_oov_u}')
    print(f'  OOV POI steps  : {val_oov_p} / {val_total_p} '
          f'({100.0*val_oov_p/max(val_total_p,1):.1f}%)')
    print(f'  OOV trajectories: {val_oov_traj} / {len(val_trajectories)} '
          f'({100.0*val_oov_traj/max(len(val_trajectories),1):.1f}%)')
    print(f'\nOOV (test):')
    print(f'  OOV users      : {test_oov_u}')
    print(f'  OOV POI steps  : {test_oov_p} / {test_total_p} '
          f'({100.0*test_oov_p/max(test_total_p,1):.1f}%)')
    print(f'  OOV trajectories: {test_oov_traj} / {len(test_trajectories)} '
          f'({100.0*test_oov_traj/max(len(test_trajectories),1):.1f}%)')
    print(f'\nCSV files:')
    for k, v in csv_paths.items():
        print(f'  {v}')
    print(f'  {graph_path}')
    print('=================================================')

    # =========================================================
    # 14. Sanity checks
    # =========================================================
    print('\n========== Sanity Checks ==========')
    all_ok = True

    # Check 1: trajectory_id.split('_')[0] == user_id_raw
    sample = random.choice(train_trajectories)
    uid_from_id = sample['trajectory_id'].split('_')[0]
    ok1 = uid_from_id == sample['user_id_raw']
    print(f'[{"OK" if ok1 else "FAIL"}] 1. traj_id.split("_")[0] == user_id_raw  '
          f'(traj_id={sample["trajectory_id"]!r}, user_id_raw={sample["user_id_raw"]!r})')
    all_ok = all_ok and ok1

    # Check 2: graph_X.csv POI count == len(pid_dict)
    ok2 = len(graph_df) == len(pid_dict)
    print(f'[{"OK" if ok2 else "FAIL"}] 2. graph_X.csv rows ({len(graph_df)}) == len(pid_dict) ({len(pid_dict)})')
    all_ok = all_ok and ok2

    # Check 3: train CSV poi_id all in pid_dict
    train_csv = pd.read_csv(csv_paths['train'])
    train_oov = train_csv['poi_id'].astype(str).apply(lambda x: x not in pid_dict).sum()
    ok3 = train_oov == 0
    print(f'[{"OK" if ok3 else "FAIL"}] 3. {ds}_train.csv OOV POIs: {train_oov} '
          f'(expected 0 — train POIs must all be in pid_dict)')
    all_ok = all_ok and ok3

    # Check 4: val/test CSV OOV POI ratios
    for split_name in ['val', 'test']:
        sdf = pd.read_csv(csv_paths[split_name])
        n_oov = sdf['poi_id'].astype(str).apply(lambda x: x not in pid_dict).sum()
        pct   = 100.0 * n_oov / max(len(sdf), 1)
        print(f'[INFO] 4. {ds}_{split_name}.csv OOV POI steps: {n_oov} / {len(sdf)} ({pct:.1f}%)')

    # Check 5: timestamps monotonically increasing within each trajectory
    n_bad = 0
    for traj in train_trajectories + val_trajectories + test_trajectories:
        ts = traj['timestamp_seq']
        if any(ts[i] > ts[i + 1] for i in range(len(ts) - 1)):
            n_bad += 1
    ok5 = n_bad == 0
    print(f'[{"OK" if ok5 else "FAIL"}] 5. Monotone timestamps: {n_bad} trajectories with violations')
    all_ok = all_ok and ok5

    print(f'\nAll checks passed: {all_ok}')
    print('===================================')
