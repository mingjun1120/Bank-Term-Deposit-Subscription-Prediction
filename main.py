import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')

def main():
    # Load data
    ingestion = Ingestion()
    data = ingestion.load_data()
    logging.info("DATA INGESTION COMPLETED SUCCESSFULLY!!!\n")

    # Clean data
    cleaner = Cleaner()
    data = cleaner.clean_data(data)
    logging.info("DATA CLEANING COMPLETED SUCCESSFULLY!!!\n")
    
    # Prepare and train model
    trainer = Trainer()
    X, y = trainer.feature_target_separator(data)
    X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
    trainer.train_model(X_train, y_train, X_test, y_test)
    trainer.save_model()
    logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY!!!\n")

    # Evaluate model
    predictor = Predictor()
    accuracy_train, precision_train, recall_train, f1_train, roc_auc_score_train, class_report_train = predictor.evaluate_model(X_train, y_train)
    accuracy, precision, recall, f1, roc_auc_score, class_report = predictor.evaluate_model(X_test, y_test)
    logging.info("MODEL EVALUATION COMPLETED SUCCESSFULLY!!!\n")
    
    # Print evaluation results for training set
    print("\n============== Model Evaluation Results (Train Set) ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score : {accuracy_train:.4f}")
    print(f"Precision Score: {precision_train:.4f}") 
    print(f"Recall Score   : {recall_train:.4f}")
    print(f"F1 Score       : {f1_train:.4f}")
    print(f"ROC AUC Score  : {roc_auc_score_train:.4f}")
    print(f"\nClassification Report:\n{class_report_train}")
    print("==================================================================\n")
    
    # Print evaluation results for testing set
    print("\n============== Model Evaluation Results (Test Set) ==============")
    print(f"Model: {trainer.model_name}")
    print(f"Accuracy Score : {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}") 
    print(f"Recall Score   : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC AUC Score  : {roc_auc_score:.4f}")
    print(f"\nClassification Report:\n{class_report}")
    print("=================================================================\n")


def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Model Training Experiment")
    
    with mlflow.start_run() as run:
        # Load data
        ingestion = Ingestion()
        data = ingestion.load_data()
        logging.info("DATA INGESTION COMPLETED SUCCESSFULLY!!!\n")

        # Clean data
        cleaner = Cleaner()
        cleaner = Cleaner()
        data = cleaner.clean_data(data)
        logging.info("DATA CLEANING COMPLETED SUCCESSFULLY!!!\n")
        
        # Prepare and train model
        trainer = Trainer()
        X, y = trainer.feature_target_separator(data)
        X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)
        params = trainer.train_model(X_train, y_train, X_test, y_test)
        trainer.save_model()
        logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY!!!\n")
        
        # Evaluate model
        predictor = Predictor()
        accuracy, precision, recall, f1, roc_auc_score, class_report = predictor.evaluate_model(X_test, y_test)
        report = classification_report(y_test, trainer.model.predict(X_test), output_dict=True)
        logging.info("MODEL EVALUATION COMPLETED SUCCESSFULLY!!!\n")
        
        # Tags 
        mlflow.set_tag('Model developer', 'lim_ming_jun')
        
        # Log metrics
        model_params = params # config['model']['params']
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc", roc_auc_score)
        # mlflow.log_metric('precision_report', report['weighted avg']['precision'])
        # mlflow.log_metric('recall_report', report['weighted avg']['recall'])
        mlflow.sklearn.log_model(trainer.model, "model")
        
        # Register the model
        model_name = "loan_risk_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        logging.info("MLflow tracking completed successfully")

        # Print evaluation results for training set
        accuracy_train, precision_train, recall_train, f1_train, roc_auc_score_train, class_report_train = predictor.evaluate_model(X_train, y_train)
        print("\n============== Model Evaluation Results (Train Set) ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score : {accuracy_train:.4f}")
        print(f"Precision Score: {precision_train:.4f}") 
        print(f"Recall Score   : {recall_train:.4f}")
        print(f"F1 Score       : {f1_train:.4f}")
        print(f"ROC AUC Score  : {roc_auc_score_train:.4f}")
        print(f"\nClassification Report:\n{class_report_train}")
        print("==================================================================\n")
        
        # Print evaluation results for testing set
        print("\n============== Model Evaluation Results (Test Set) ==============")
        print(f"Model: {trainer.model_name}")
        print(f"Accuracy Score : {accuracy:.4f}")
        print(f"Precision Score: {precision:.4f}") 
        print(f"Recall Score   : {recall:.4f}")
        print(f"F1 Score       : {f1:.4f}")
        print(f"ROC AUC Score  : {roc_auc_score:.4f}")
        print(f"\nClassification Report:\n{class_report}")
        print("=================================================================\n")
    

if __name__ == "__main__":
    main()
    # train_with_mlflow()