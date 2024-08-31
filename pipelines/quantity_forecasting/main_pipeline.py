from pipelines.quantity_forecasting import data_preparation_pipeline
from pipelines.quantity_forecasting import model_building_pipeline

def main():

    print ("Main pipeline started")
    
    train_features,train_target,test_features,test_target = data_preparation_pipeline.pipeline()
    model_building_pipeline.pipeline(train_features,train_target,test_features,test_target)
   
    print ("Main pipeline ended")

if __name__ == "__main__":
    main()