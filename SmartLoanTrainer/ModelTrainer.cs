using Microsoft.ML;

namespace SmartLoanTrainer
{
    public class ModelTrainer
    {
        private static readonly string dataPath = "loan_data.csv";
        private static readonly string modelPath = Path.Combine("..", "smartloanapi", "model.zip");

        public static void Train()
        {
            var mlContext = new MLContext();
            Console.WriteLine("ML.NET context created.");

            // Load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<LoanInput>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1, seed: 1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;
            Console.WriteLine("Data loaded and split.");

            // Build pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "JobTypeKey", inputColumnName: "JobType")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "JobTypeEncoded", inputColumnName: "JobTypeKey"))
                .Append(mlContext.Transforms.Concatenate("Features", "MonthlyIncome", "LoanAmount", "ReturnTime", "Age", "JobTypeEncoded"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "IsApproved", featureColumnName: "Features"));

            Console.WriteLine("Training pipeline created.");

            // Train model
            var model = pipeline.Fit(trainData);
            Console.WriteLine("Model training complete.");

            // Evaluate model
            var predictions = model.Transform(testData);
            
            try 
            {
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "IsApproved");

                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
                Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            }
            catch (ArgumentOutOfRangeException)
            {
                Console.WriteLine("Warning: Cannot calculate metrics - test set may not contain both positive and negative samples.");
                Console.WriteLine("This is expected with very small datasets. The model has been trained successfully.");
            }

            // Save model
            string? modelDirectory = Path.GetDirectoryName(modelPath);
            if (!string.IsNullOrEmpty(modelDirectory) && !Directory.Exists(modelDirectory))
            {
                Directory.CreateDirectory(modelDirectory);
                Console.WriteLine($"Created directory: {modelDirectory}");
            }

            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
                Console.WriteLine($"Existing model file deleted: {modelPath}");
            }

            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
        }
    }

}