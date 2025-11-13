using Microsoft.ML;

namespace SmartLoanTrainer
{
    /// <summary>
    /// Main class responsible for training a machine learning model to predict loan approval
    /// Uses ML.NET framework for binary classification (approved/rejected)
    /// </summary>
    public class ModelTrainer
    {
        // File paths for data input and model output
        private static readonly string dataPath = "loan_data_realistic.csv";  // CSV file containing training data
        private static readonly string modelPath = Path.Combine("..", "smartloanapi", "model.zip");  // Where to save the trained model

        /// <summary>
        /// Main training method that handles the entire ML pipeline:
        /// 1. Load and analyze data
        /// 2. Split data into train/test sets
        /// 3. Create ML pipeline with transformations
        /// 4. Train the model
        /// 5. Evaluate performance
        /// 6. Save the model for use in the API
        /// </summary>
        public static void Train()
        {
            // Create ML.NET context - this is the starting point for all ML.NET operations
            // It provides access to all ML algorithms and data transformations
            var mlContext = new MLContext();
            Console.WriteLine("ML.NET context created.");

            #region Data Loading and Analysis
            // Load data from CSV file into ML.NET's IDataView format
            // IDataView is ML.NET's main data structure for handling datasets efficiently
            IDataView dataView = mlContext.Data.LoadFromTextFile<LoanInput>(
                path: dataPath,           // Path to the CSV file
                hasHeader: true,          // First row contains column names
                separatorChar: ',');      // Comma-separated values

            // Analyze the data distribution before training to detect potential issues
            // This helps identify problems like perfect correlation that lead to unrealistic results
            AnalyzeDataDistribution(dataView, mlContext);
            #endregion

            #region Data Splitting
            // Split data into training (80%) and testing (20%) sets
            // Training set: used to train the model
            // Test set: used to evaluate how well the model performs on unseen data
            // seed=1 ensures reproducible results across runs
            var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;
            Console.WriteLine("Data loaded and split.");
            #endregion

            #region ML Pipeline Creation
            // Create a machine learning pipeline - a series of data transformations and algorithms
            // Each step in the pipeline processes data and passes it to the next step
            var pipeline =
                // STEP 1: Convert JobType string to numeric key (required for ML algorithms)
                // Example: "Full-time" -> 1, "Part-time" -> 2, "Student" -> 3, "Unemployed" -> 4
                mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "JobTypeKey", inputColumnName: "JobType")

                // STEP 2: One-hot encode the job type keys into binary features
                // Example: JobType=2 becomes [0,1,0,0] vector
                // This prevents the algorithm from assuming ordinal relationships between job types
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "JobTypeEncoded", inputColumnName: "JobTypeKey"))

                // STEP 3: Combine all features into a single "Features" column
                // ML algorithms need all input features in one column
                .Append(mlContext.Transforms.Concatenate("Features", "MonthlyIncome", "LoanAmount", "ReturnTime", "Age", "JobTypeEncoded"))

                // STEP 4: Normalize features to same scale (0-1 range)
                // Prevents features with large values (like LoanAmount) from dominating smaller ones (like Age)
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))

                // STEP 5: Apply the actual machine learning algorithm - Logistic Regression
                // This algorithm is good for binary classification and less prone to overfitting
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "IsApproved",    // What we're trying to predict
                    featureColumnName: "Features",    // Input features
                    l2Regularization: 1.0f));        // Regularization prevents overfitting

            Console.WriteLine("Training pipeline created.");
            #endregion

            #region Model Training
            // Train the model using the training data
            // This process finds the best weights/parameters to make accurate predictions
            var model = pipeline.Fit(trainData);
            Console.WriteLine("Model training complete.");
            #endregion

            #region Model Evaluation
            // Test the model's performance on data it has never seen before
            var predictions = model.Transform(testData);

            try
            {
                // Calculate performance metrics
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "IsApproved");

                // Display key performance indicators:
                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");           // % of correct predictions
                Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");       // How well model ranks positive vs negative
                Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");           // Balance of precision and recall

                // Detailed analysis of test results to understand model behavior
                AnalyzeTestResults(testData, predictions, mlContext);
            }
            catch (ArgumentOutOfRangeException)
            {
                // Handle edge case where test set doesn't contain both positive and negative examples
                Console.WriteLine("Warning: Cannot calculate metrics - test set may not contain both positive and negative samples.");
                Console.WriteLine("This is expected with very small datasets. The model has been trained successfully.");
            }
            #endregion

            #region Model Saving
            // Save the trained model to disk so the API can use it for predictions
            string? modelDirectory = Path.GetDirectoryName(modelPath);

            // Create directory if it doesn't exist
            if (!string.IsNullOrEmpty(modelDirectory) && !Directory.Exists(modelDirectory))
            {
                Directory.CreateDirectory(modelDirectory);
                Console.WriteLine($"Created directory: {modelDirectory}");
            }

            // Delete existing model file to avoid conflicts
            if (File.Exists(modelPath))
            {
                File.Delete(modelPath);
                Console.WriteLine($"Existing model file deleted: {modelPath}");
            }

            // Save the model with its data schema for later use
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
            #endregion
        }        /// <summary>
                 /// Analyzes the distribution of data to identify potential problems before training
                 /// This is crucial for detecting issues like perfect correlation that lead to unrealistic results
                 /// </summary>
                 /// <param name="dataView">The loaded dataset</param>
                 /// <param name="mlContext">ML.NET context for data operations</param>
        private static void AnalyzeDataDistribution(IDataView dataView, MLContext mlContext)
        {
            Console.WriteLine("\n=== DATA ANALYSIS ===");

            // Convert ML.NET IDataView to regular C# list for easier analysis
            // reuseRowObject: false ensures each row gets its own object (safer for analysis)
            var data = mlContext.Data.CreateEnumerable<LoanInput>(dataView, reuseRowObject: false);
            var dataList = data.ToList();

            Console.WriteLine($"Total records: {dataList.Count}");

            // Group data by job type and calculate approval statistics
            // This helps identify if certain job types have 100% approval/rejection rates
            var jobTypeAnalysis = dataList.GroupBy(x => x.JobType)
                .Select(g => new
                {
                    JobType = g.Key,                                                    // Job type name
                    Total = g.Count(),                                                  // Total applications for this job type
                    Approved = g.Count(x => x.IsApproved),                             // Number approved
                    Rejected = g.Count(x => !x.IsApproved),                           // Number rejected
                    ApprovalRate = g.Count(x => x.IsApproved) / (double)g.Count() * 100  // Approval percentage
                });

            Console.WriteLine("\nApproval rates by job type:");
            foreach (var analysis in jobTypeAnalysis)
            {
                Console.WriteLine($"{analysis.JobType}: {analysis.ApprovalRate:F1}% ({analysis.Approved}/{analysis.Total})");
            }

            // Check for perfect correlation - this is a major red flag!
            // If any job type has 0% OR 100% approval, the model can cheat by only looking at job type
            bool perfectCorrelation = jobTypeAnalysis.All(x => x.ApprovalRate == 0 || x.ApprovalRate == 100);
            if (perfectCorrelation)
            {
                Console.WriteLine("\n⚠️  WARNING: Perfect correlation detected between JobType and approval!");
                Console.WriteLine("   This will result in 100% accuracy but indicates unrealistic data.");
                Console.WriteLine("   Consider adding more diverse, realistic examples.");
            }

            // Show overall approval rate for context
            Console.WriteLine("\nOverall approval rate: {0:F1}%",
                dataList.Count(x => x.IsApproved) / (double)dataList.Count * 100);
            Console.WriteLine("=====================\n");
        }

        /// <summary>
        /// Analyzes the test results in detail to understand model performance and behavior
        /// This helps identify issues like overfitting or unrealistic confidence scores
        /// </summary>
        /// <param name="testData">The test dataset</param>
        /// <param name="predictions">Model predictions on test data</param>
        /// <param name="mlContext">ML.NET context for data operations</param>
        private static void AnalyzeTestResults(IDataView testData, IDataView predictions, MLContext mlContext)
        {
            Console.WriteLine("\n=== TEST SET ANALYSIS ===");

            // Convert predictions and test data to C# objects for analysis
            var predictionResults = mlContext.Data.CreateEnumerable<PredictionResult>(predictions, reuseRowObject: false).ToList();
            var testDataList = mlContext.Data.CreateEnumerable<LoanInput>(testData, reuseRowObject: false).ToList();

            // Show basic test set composition
            Console.WriteLine($"Test set size: {testDataList.Count}");
            Console.WriteLine($"Positive samples: {testDataList.Count(x => x.IsApproved)}");    // How many approved loans
            Console.WriteLine($"Negative samples: {testDataList.Count(x => !x.IsApproved)}");   // How many rejected loans

            // Display sample predictions with probability scores
            // Probability scores show how confident the model is (0.0 = certain rejection, 1.0 = certain approval)
            Console.WriteLine("\nSample predictions (showing probability scores):");
            for (int i = 0; i < Math.Min(10, predictionResults.Count); i++)
            {
                var actual = testDataList[i];        // What actually happened
                var prediction = predictionResults[i]; // What the model predicted

                Console.WriteLine($"JobType: {actual.JobType,-12} Income: {actual.MonthlyIncome,5:F0} " +
                                $"LoanAmt: {actual.LoanAmount,6:F0} Actual: {actual.IsApproved,-5} " +
                                $"Predicted: {prediction.PredictedLabel,-5} Probability: {prediction.Probability:F3}");
            }

            // Analyze probability separation - this is key for understanding AUC scores
            // Get all probabilities for approved loans (should be higher)
            var positiveProbabilities = predictionResults.Where((p, i) => testDataList[i].IsApproved).Select(p => p.Probability);
            // Get all probabilities for rejected loans (should be lower)
            var negativeProbabilities = predictionResults.Where((p, i) => !testDataList[i].IsApproved).Select(p => p.Probability);

            if (positiveProbabilities.Any() && negativeProbabilities.Any())
            {
                var minPositive = positiveProbabilities.Min();  // Lowest probability among approved loans
                var maxNegative = negativeProbabilities.Max();  // Highest probability among rejected loans

                Console.WriteLine($"\nProbability separation:");
                Console.WriteLine($"Lowest positive probability: {minPositive:F3}");
                Console.WriteLine($"Highest negative probability: {maxNegative:F3}");

                // If all approved loans have higher probabilities than all rejected loans,
                // this creates perfect separation and 100% AUC (which might indicate overfitting)
                if (minPositive >= maxNegative)
                {
                    Console.WriteLine("⚠️  Perfect separation detected! All positive probabilities > all negative probabilities");
                    Console.WriteLine("   This explains the 100% AUC. Consider adding more challenging cases.");
                }
            }
            Console.WriteLine("========================\n");
        }

        /// <summary>
        /// Data structure to hold model predictions
        /// This matches the output format of ML.NET binary classification models
        /// </summary>
        public class PredictionResult
        {
            public bool PredictedLabel { get; set; }    // True = approved, False = rejected
            public float Probability { get; set; }      // Confidence score (0.0 to 1.0)
        }
    }

}