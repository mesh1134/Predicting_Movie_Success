import joblib

def inspect_model(filename):
    print(f"Inspecting {filename}:")
    try:
        obj = joblib.load(filename)
        print(f"  Type: type(obj)")
        
        if hasattr(obj, 'feature_names_in_'):
            print(f"  Features: {obj.feature_names_in_}")
        elif hasattr(obj, 'get_feature_names_out'):
            try:
                print(f"  Feature names out: {obj.get_feature_names_out()}")
            except Exception as e:
                print(f"  Could not get feature names out: {e}")
                
        if hasattr(obj, 'classes_'):
            print(f"  Classes: {obj.classes_}")
            
    except Exception as e:
        print(f"  Error loading: {e}")

inspect_model('models/best_movie_model.pkl')
inspect_model('models/best_movie_model_xgboost.pkl')
inspect_model('models/ordinal_encoder.pkl')
inspect_model('models/standard_scaler.pkl')
inspect_model('models/target_label_encoder.pkl')
