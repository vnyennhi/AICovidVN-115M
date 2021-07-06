# import all dependencies
from utils import *
import warnings
warnings.filterwarnings("ignore")


# update file paths accordingly
test_dataset_path = 'private_test_audio_files_8k'
test_metadata = pd.read_csv('metadata/metadata_private_test.csv')

encoder_path = 'model/encoder.pickle'
model_path = 'model/xgb_model.pickle'


print("Extracting features for test set")

# extract sound features for test set
test_sound_features = extract_sound_features(test_metadata, test_dataset_path)

# extract metadata features for both datasets
test_metadata_features = extract_metadata_features(test_metadata, encoder_path)

# concatenate into numpy arrays
test = np.concatenate([np.array(test_sound_features), test_metadata_features], axis=1)

print("Load trained model and output prediction")

# load model for prediction
with open(model_path, 'rb') as f:
    xg = pickle.load(f)

# predict on test set
pred = xg.predict_proba(test)


# prepare and save predictions for submission
y_preds = pd.DataFrame(test_metadata['uuid'],columns=['uuid'])
y_preds['assessment_result'] =  np.array(pred)[:,1]
y_preds.to_csv('results.csv', index=False)

print("Finished")
