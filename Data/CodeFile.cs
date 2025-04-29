using CategoryPrecdictAI.DataStructures;
using Microsoft.ML;

namespace JML
{
    public static class test
    {
        public static void PredictItem(MLContext mlContext, string itemName, string categoryModelPath, string departmentModelPath)
        {
            // Load the models
            ITransformer loadedCategoryModel = mlContext.Model.Load(categoryModelPath, out _);
            ITransformer loadedDepartmentModel = mlContext.Model.Load(departmentModelPath, out _);

            //// Create prediction engines
            //// IMPORTANT: Input is ItemData, Output is the specific Prediction class
            //var categoryPredEngine = mlContext.Model.CreatePredictionEngine<ItemData, CategoryPrediction>(loadedCategoryModel);
            //var departmentPredEngine = mlContext.Model.CreatePredictionEngine<ItemData, DepartmentPrediction>(loadedDepartmentModel);

            //// Create input data
            //var inputData = new ItemData { ItemName = itemName };

            //// Make predictions
            //var categoryPrediction = categoryPredEngine.Predict(inputData);
            //var departmentPrediction = departmentPredEngine.Predict(inputData);

            // --- Category Prediction using Transform ---
            // 1. Create a single-item IDataView from the input data
            var categoryInputList = new List<ItemData>() { new ItemData { ItemName = itemName } };
            IDataView categoryInputView = mlContext.Data.LoadFromEnumerable(categoryInputList);

            // 2. Transform the input data
            IDataView categoryPredictionView = loadedCategoryModel.Transform(categoryInputView);

            // 3. Extract the prediction
            // Use CreateEnumerable with the correct prediction class. Set reuseRowObject to false for safety with single items.
            CategoryPrediction categoryPredictionResult = mlContext.Data
                                                                .CreateEnumerable<CategoryPrediction>(categoryPredictionView, reuseRowObject: false)
                                                                .First(); // Get the first (and only) prediction

            // --- Department Prediction using Transform ---
            // 1. Create a single-item IDataView (can reuse ItemData instance if desired)
            var departmentInputList = new List<ItemData>() { new ItemData { ItemName = itemName } }; // Or just use the same list if ItemData is sufficient for both models' input schema
            IDataView departmentInputView = mlContext.Data.LoadFromEnumerable(departmentInputList);

            // 2. Transform the input data
            IDataView departmentPredictionView = loadedDepartmentModel.Transform(departmentInputView);

            // 3. Extract the prediction
            DepartmentPrediction departmentPredictionResult = mlContext.Data
                                                                    .CreateEnumerable<DepartmentPrediction>(departmentPredictionView, reuseRowObject: false)
                                                                    .First(); // Get the first prediction

            Console.WriteLine($"--- Prediction for: '{itemName}' ---");
            Console.WriteLine($"   Predicted Category:   {categoryPredictionResult.PredictedCategory}");
            Console.WriteLine($"   Predicted Department: {departmentPredictionResult.PredictedDepartment}");
            Console.WriteLine($"--------------------------------------");
        }

    }
    

}
