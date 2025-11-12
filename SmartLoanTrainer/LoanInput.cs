using Microsoft.ML.Data;

namespace SmartLoanTrainer
{
    public class LoanInput
    {
        [LoadColumn(0)]
        public float MonthlyIncome { get; set; }

        [LoadColumn(1)]
        public float LoanAmount { get; set; }

        [LoadColumn(2)]
        public float ReturnTime { get; set; }

        [LoadColumn(3)]
        public float Age { get; set; }

        [LoadColumn(4)]
        public string JobType { get; set; } = string.Empty;

        [LoadColumn(5)]
        public bool IsApproved { get; set; } // This is the label
    }
}