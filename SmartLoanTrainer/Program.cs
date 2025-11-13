using System;
using Microsoft.ML;
using SmartLoanTrainer;

namespace SmartLoanTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Enter 1 to train the loan model, or 2 to test model.zip:");
            string input = Console.ReadLine();

            if (input == "1")
            {
                Console.WriteLine("Starting loan model training...");
                ModelTrainer.Train();
                Console.WriteLine("Training complete. Model saved as model.zip");
            }
            else if (input == "2")
            {
                var mlContext = new MLContext();

                // Load the model
                ITransformer model = mlContext.Model.Load("model.zip", out var inputSchema);

                // Create prediction engine
                var predictionEngine = mlContext.Model.CreatePredictionEngine<LoanInput, LoanPrediction>(model);

                // Sample input
                var sample = new LoanInput
                {
                    MonthlyIncome = 40500,
                    LoanAmount = 20000,
                    ReturnTime = 24,
                    Age = 35,
                    JobType = "Full-time"
                };

                // Run prediction
                var result = predictionEngine.Predict(sample);
                Console.WriteLine("Loan Prediction for sample input:");
                Console.WriteLine($"Monthly Income: {sample.MonthlyIncome}");
                Console.WriteLine($"Loan Amount: {sample.LoanAmount}");
                Console.WriteLine($"Return Time (months): {sample.ReturnTime}");
                Console.WriteLine($"Age: {sample.Age}");
                Console.WriteLine($"Job Type: {sample.JobType}");
                Console.WriteLine("Loan Prediction Result:");
                Console.WriteLine($"Prediction: {(result.Prediction ? "✅ Approved" : "❌ Rejected")}");
                Console.WriteLine($"Confidence: {result.Probability:P2}");
            }
            else
            {
                Console.WriteLine("Invalid choice. Please enter 1 or 2.");
            }
        }
    }
}