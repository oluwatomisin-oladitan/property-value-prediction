import json
import zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# load the data
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
train_records = load_json('train.json')
test_records = load_json('test.json')

print(f'Traing examples: {len(train_records)}')
print(f'Test examples: {len(test_records)}')

# Compute median property value per commune and per property type

def build_target_encodings(train_records):
    commune_vals = defaultdict(list)
    pt_vals = defaultdict(list)

    for r in train_records:
        v = r['property_value']
        commune_vals[r['commune_codes']].append(v)
        pt_vals[r['property_type']].append(v)

    commune_median = {k: float(np.median(v)) for k,v in commune_vals.items()}
    pt_median = {k: float(np.median(v)) for k, v in pt_vals.items()}
    global_median = float(np.median([r['property_value'] for r in train_records]))

    # Count transactions per commune
    commune_count = {k: len(v) for k, v in commune_vals.items()}

    # Median per property type AND commune combined
    combined_vals = defaultdict(list)
    for r in train_records:
        key = r['commune_codes'] + '_' + r['property_type']
        combined_vals[key].append(r['property_value'])
    
    combined_median = {k: float(np.median(v)) for k, v in combined_vals.items()}
    
    return commune_median, pt_median, global_median, commune_count, combined_median

commune_median, pt_median, global_median, commune_count, combined_median = build_target_encodings(train_records)
print(f'Global medain property value: {global_median:,.0f}')

# Label encode transaction types

def transaction_type_map(train_records):
    all_types = []
    for r in train_records:
        all_types.append(r['transaction_type'])
    unique_types = sorted(set(all_types))
    tt_map ={}
    for i,v in enumerate(unique_types):
        tt_map[v] = i
    return tt_map
            
tt_map = transaction_type_map(train_records)

print(f'Transaction types found: {list(tt_map.keys())}')


# Extract features from records
def extract_features(records, commune_median, pt_median, global_median, tt_map, commune_count, combined_median):
    rows = []
    for r in records:
        row = {}

        # Numeric features
        numeric = ['built_area', 'num_lots', 'num_commercial', 'land_area',
                   'num_premises', 'num_houses', 'num_apartments', 'num_dependencies',
                   'house_area', 'apartment_area', 'num_parcels', 'num_sections',
                   'num_communes', 'year', 'month',
                   'num_apt_1_room', 'num_apt_2_rooms', 'num_apt_3_rooms',
                   'num_apt_4_rooms', 'num_apt_5plus_rooms',
                   'num_house_1_room', 'num_house_2_rooms', 'num_house_3_rooms',
                   'num_house_4_rooms', 'num_house_5plus_rooms',
                   'area_apt_1_room', 'area_apt_2_rooms', 'area_apt_3_rooms',
                   'area_apt_4_rooms', 'area_apt_5plus_rooms',
                   'area_house_1_room', 'area_house_2_rooms', 'area_house_3_rooms',
                   'area_house_4_rooms', 'area_house_5plus_rooms']

        for key in numeric:
            row[key] = float(r.get(key, 0) or 0)

        # Boolean feature
        if r.get('future_sale'):
            row['future_sale'] = 1.0 
        else:
            row['future_sale'] = 0.0

        # Label encoded
        row['transaction_type'] = float(tt_map.get(r.get('transaction_type', ''), -1))

        # Target encoding
        row['commune_median_value'] = commune_median.get(r.get('commune_codes', ''), global_median)
        row['property_type_median'] = pt_median.get(r.get('property_type', ''), global_median)

        # Engineered features
        built = float(r.get('built_area', 0) or 0)
        land = float(r.get('land_area', 0) or 0)
        total_rooms = sum(float(r.get(k, 0) or 0) for k in [
            'num_apt_1_room', 'num_apt_2_rooms', 'num_apt_3_rooms',
            'num_apt_4_rooms', 'num_apt_5plus_rooms',
            'num_house_1_room', 'num_house_2_rooms', 'num_house_3_rooms',
            'num_house_4_rooms', 'num_house_5plus_rooms'])

        row['total_rooms'] = total_rooms
        row['total_area'] = built + land
        row['built_area_sq'] = built * built
        row['area_per_room'] = built / max(total_rooms, 1)
        row['land_to_built_ratio'] = land / max(built, 1)
        row['num_total_properties'] = float(r.get('num_houses', 0) or 0) + float(r.get('num_apartments', 0) or 0)
        # More engineered features
        row['has_house'] = 1.0 if float(r.get('num_houses', 0) or 0) > 0 else 0.0
        row['has_apartment'] = 1.0 if float(r.get('num_apartments', 0) or 0) > 0 else 0.0
        row['has_commercial'] = 1.0 if float(r.get('num_commercial', 0) or 0) > 0 else 0.0
        row['land_area_sq'] = land * land
        row['house_area_sq'] = float(r.get('house_area', 0) or 0) ** 2

        row['commune_transaction_count'] = float(commune_count.get(r.get('commune_codes', ''), 0))
        combined_key = r.get('commune_codes', '') + '_' + r.get('property_type', '')
        row['commune_property_type_median'] = combined_median.get(combined_key, global_median)
        row['price_per_built_sqm'] = commune_median.get(r.get('commune_codes', ''), global_median) / max(built, 1)
        row['price_per_land_sqm'] = commune_median.get(r.get('commune_codes', ''), global_median) / max(land, 1)
        rows.append(row)

    return pd.DataFrame(rows)

# Prepare features for training and 

print('Extracting features...')

X_all = extract_features(train_records, commune_median, pt_median, global_median, tt_map, commune_count, combined_median)
y_all = np.array([r['property_value'] for r in train_records], dtype=np.float64)

print(f'Training examples: {len(y_all)}')
X_test = extract_features(test_records, commune_median, pt_median, global_median, tt_map, commune_count, combined_median)

print(f'Training feature shape: {X_all.shape}')
print(f'Test feature shape: {X_test.shape}')

y_all_log = np.log1p(y_all)
print(f'Training on full dataset: {len(X_all)} examples')

# Train the model
print('Training model...')
model = lgb.LGBMRegressor(
    n_estimators=2489,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=255,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective='regression_l1',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_all, y_all_log)



# Generate predictions for test set
print('Generating predictions...')
test_predictions = np.expm1(model.predict(X_test))

# Save predictions to predicted.json
predictions = [{'property_value': float(v)} for v in test_predictions]
with open('predicted.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=2)

# Zip it up for submission
with zipfile.ZipFile('predicted.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    archive.write('predicted.json', arcname='predicted.json')

print(f'Done! Wrote {len(predictions)} predictions to predicted.zip')