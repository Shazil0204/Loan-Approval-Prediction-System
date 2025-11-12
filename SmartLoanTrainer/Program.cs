namespace SmartLoanTrainer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting loan model training...");
            ModelTrainer.Train();
            Console.WriteLine("Training complete. Model saved as model.zip");
        }
    }
}