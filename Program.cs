using CategoryPrecdictAI.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq; // Required for LINQ operations if used

// Assume ItemData, CategoryPrediction, DepartmentPrediction classes are defined as above

public class Program
{
    // --- Configuration ---
    // Adjust paths as needed
    private static readonly string BaseDataPath = Path.Combine(Environment.CurrentDirectory, "Data");
    private static readonly string TrainDataPath = Path.Combine(BaseDataPath, "data.csv"); // Your training data file
    private static readonly string CategoryModelPath = Path.Combine(Environment.CurrentDirectory, "category_model.zip");
    private static readonly string DepartmentModelPath = Path.Combine(Environment.CurrentDirectory, "department_model.zip");

    private static readonly bool isTraining = true; // Set to false if you want to skip training and just load the model
    public static void Main(string[] args)
    {
        // --- Configure Logging ---
        // Create a logger factory that sends logs to the console
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddFilter("Microsoft.ML", LogLevel.Information) // Show Information level logs from ML.NET components
                                                                 // You can adjust the LogLevel:
                                                                 // LogLevel.Trace -> Very detailed (might include per-instance details)
                                                                 // LogLevel.Debug -> Detailed debug info
                                                                 // LogLevel.Information -> Standard progress (like trainer iterations) - GOOD STARTING POINT
                                                                 // LogLevel.Warning -> Only warnings and errors
                                                                 // LogLevel.Error -> Only errors
                                                                 // LogLevel.Critical -> Only critical failures
                                                                 // LogLevel.None -> Nothing
                .AddConsole(); // Add the console logger provider
        });

        // Initialize MLContext
        var mlContext = new MLContext(seed: 0); // Seed for reproducibility

        

        if (isTraining)
        {
            Console.WriteLine("Loading data...");
            // Load data from CSV. Adjust separatorChar and hasHeader if needed.
            IDataView dataView = mlContext.Data.LoadFromTextFile<ItemData>(
                path: TrainDataPath,
                separatorChar: ',', // Use '\t' for TSV
                hasHeader: false);   // Set to true if your file has a header row

            // --- Data Preprocessing & Splitting ---
            // It's crucial to split data for evaluation. 80% training, 20% testing is common.
            // Stratified split is often better for classification but requires more setup.
            // Basic random split:
            DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 0);
            IDataView trainingData = trainTestData.TrainSet;
            IDataView testingData = trainTestData.TestSet; // Use this for evaluation

            // --- Build & Train Category Model ---
            Console.WriteLine("\nBuilding and Training Category Model...");
            var categoryStopwatch = Stopwatch.StartNew(); // Start timer
            var categoryModel = TrainCategoryModel(mlContext, trainingData);
            categoryStopwatch.Stop(); // Stop timer
            Console.WriteLine($"Category Model training finished in: {categoryStopwatch.ElapsedMilliseconds} ms");

            // --- Evaluate Category Model ---
            Console.WriteLine("\nEvaluating Category Model...");
            EvaluateModel(mlContext, categoryModel, testingData, "Category");

            // --- Save Category Model ---
            Console.WriteLine("\nSaving Category Model...");
            mlContext.Model.Save(categoryModel, trainingData.Schema, CategoryModelPath);
            Console.WriteLine($"Category Model saved to: {CategoryModelPath}");

            // --- Build & Train Department Model ---
            Console.WriteLine("\nBuilding and Training Department Model...");
            var departmentStopwatch = Stopwatch.StartNew(); // Start timer
            var departmentModel = TrainDepartmentModel(mlContext, trainingData);
            departmentStopwatch.Stop(); // Stop timer
            Console.WriteLine($"Department Model training finished in: {departmentStopwatch.ElapsedMilliseconds} ms");

            // --- Evaluate Department Model ---
            Console.WriteLine("\nEvaluating Department Model...");
            EvaluateModel(mlContext, departmentModel, testingData, "Department");

            // --- Save Department Model ---
            Console.WriteLine("\nSaving Department Model...");
            mlContext.Model.Save(departmentModel, trainingData.Schema, DepartmentModelPath);
            Console.WriteLine($"Department Model saved to: {DepartmentModelPath}");
        }
       

        // --- Example Prediction ---
        Console.WriteLine("\n--- Making Example Predictions ---");
        //PredictItem(mlContext, "FIRE WOOD BAG", CategoryModelPath, DepartmentModelPath);
        //PredictItem(mlContext, "PEANUT 400gm", CategoryModelPath, DepartmentModelPath);
        //PredictItem(mlContext, "SMINT MINT 8GM", CategoryModelPath, DepartmentModelPath);

        bool isExistRequested = false;

        while(!isExistRequested)
        {
            Console.WriteLine("Enter item name to get category and department...\nOr Enter 'Exit' to exit ");
            var key = Console.ReadLine();
            if (key == "Exit")
            {
                isExistRequested = true;
            }

            PredictItem (mlContext, key, CategoryModelPath, DepartmentModelPath);
        }
    }

    // --- Training Method for Category ---
    private static ITransformer TrainCategoryModel(MLContext mlContext, IDataView trainingData)
    {
        Console.WriteLine("Defining Category pipeline..."); // Manual message
        // Define the training pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(ItemData.Category), outputColumnName: "Label") // Keep Label as the key column name
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(ItemData.ItemName), outputColumnName: "Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")) // Trainer outputs 'PredictedLabel' (key) by default                                                                                                        
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedCategoryValue", inputColumnName: "PredictedLabel"));

        Console.WriteLine("Starting Category model training (watch for ML.NET logs)..."); // Manual message
        var model = pipeline.Fit(trainingData);
        Console.WriteLine("Category model training complete."); // Manual message
        return model;
    }

    // --- Training Method for Department ---
    private static ITransformer TrainDepartmentModel(MLContext mlContext, IDataView trainingData)
    {
        Console.WriteLine("Defining Department pipeline..."); // Manual message
        // Define the training pipeline (similar to category, but maps Department)
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(ItemData.Department), outputColumnName: "Label") // Keep Label as the key column name
            .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(ItemData.ItemName), outputColumnName: "Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")) // Trainer outputs 'PredictedLabel' (key) by default
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedDepartmentValue", inputColumnName: "PredictedLabel"));

        Console.WriteLine("Starting Department model training (watch for ML.NET logs)..."); // Manual message
        var model = pipeline.Fit(trainingData);
        Console.WriteLine("Department model training complete."); // Manual message
        return model;
    }

    // --- Evaluation Method (Generic) ---
    private static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testData, string labelColumnName)
    {
        Console.WriteLine($"Evaluating model for {labelColumnName}...");
        var predictions = model.Transform(testData);

        // Use the specific evaluator for Multiclass Classification
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score"); // Ensure 'Label' matches the output of MapValueToKey

        Console.WriteLine($" * Metrics for {labelColumnName} model *");
        Console.WriteLine($"   - MicroAccuracy:    {metrics.MicroAccuracy:P2}"); // Overall accuracy
        Console.WriteLine($"   - MacroAccuracy:    {metrics.MacroAccuracy:P2}"); // Average accuracy per class (good for imbalanced data)
        Console.WriteLine($"   - LogLoss:          {metrics.LogLoss:#.###}");     // Lower is better
        Console.WriteLine($"   - LogLossReduction: {metrics.LogLossReduction:#.###}"); // Closer to 1 is better

        // You can print the confusion matrix for more detail if needed (can be large)
        // Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }

    // --- Prediction Method ---
    private static void PredictItem(MLContext mlContext, string itemName, string categoryModelPath, string departmentModelPath)
    {
        // Load the models
        ITransformer loadedCategoryModel = mlContext.Model.Load(categoryModelPath, out var categoryModelSchema);
        ITransformer loadedDepartmentModel = mlContext.Model.Load(departmentModelPath, out var departmentModelSchema);

        // Create prediction engines
        // IMPORTANT: Input is ItemData, Output is the specific Prediction class
        var categoryPredEngine = mlContext.Model.CreatePredictionEngine<ItemData, CategoryPrediction>(loadedCategoryModel);
        var departmentPredEngine = mlContext.Model.CreatePredictionEngine<ItemData, DepartmentPrediction>(loadedDepartmentModel);

        // Create input data
        var inputData = new ItemData { ItemName = itemName };

        // Make predictions
        var categoryPrediction = categoryPredEngine.Predict(inputData);
        var departmentPrediction = departmentPredEngine.Predict(inputData);

        Console.WriteLine($"--- Prediction for: '{itemName}' ---");
        Console.WriteLine($"   Predicted Category:   {categoryPrediction.PredictedCategory} , score : {categoryPrediction.Score.ToString()}");
        Console.WriteLine($"   Predicted Department: {departmentPrediction.PredictedDepartment} , score : {departmentPrediction.Score.ToString()}");
        Console.WriteLine($"--------------------------------------");
    }
}