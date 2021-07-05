# import all dependencies
from utils import *


# update file paths accordingly
train_dataset_path = 'train_audio_files_8k'
train_metadata = pd.read_csv('metadata/metadata_train_challenge.csv')

test_dataset_path = 'private_test_audio_files_8k'
test_metadata = pd.read_csv('metadata/metadata_private_test.csv')

encoder_path = 'model/encoder.pickle'
model_path = 'model/xgb_model.pickle'

train_npy_path = 'model/X_train.npy'
test_npy_path = 'model/X_test.npy'


# extract sound features for train set
train_sound_features = extract_sound_features(train_metadata, train_dataset_path)

# extract sound features for test set
test_sound_features = extract_sound_features(test_metadata, test_dataset_path)

# extract metadata features for both datasets
train_metadata_features = extract_metadata_features(train_metadata, encoder_path)
test_metadata_features = extract_metadata_features(test_metadata, encoder_path)

# concatenate into numpy arrays
train = np.concatenate([np.array(train_sound_features), train_metadata_features], axis=1)
test = np.concatenate([np.array(test_sound_features), test_metadata_features], axis=1)


# train model
xg = xgb.XGBClassifier(max_depth=7,learning_rate=0.07,
                     n_estimators=200,
                     silent=1,eta=1,objective='binary:logistic',
                     num_round=50, eval_metric='auc')  
xg.fit(train,train_metadata['assessment_result'])

# predict on test set
pred = xg.predict_proba(test)


# prepare and save predictions for submission
y_preds = pd.DataFrame(test_metadata['uuid'],columns=['uuid'])
y_preds['assessment_result'] =  np.array(pred)[:,1]
y_preds.to_csv('results.csv', index=False)


# save other artifacts: model and feature arrays of train/test sets
with open(model_path, 'wb') as f:
    pickle.dump(xg, f)
with open(train_npy_path, 'wb') as f:
    pickle.dump(train, f)
with open(test_npy_path, 'wb') as f:
    pickle.dump(test, f)