using Microsoft.ML.Data;

namespace SmartLoanTrainer
{
    public class LoanPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}