from pyspark.ml.image import ImageSchema
import pyspark.sql.functions as f
import pyspark.sql.functions 
from pyspark.sql.functions import col, when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys

def main():
    
    #get or create spark session to read image and json data
    spark = SparkSession.builder.appName("yelp").getOrCreate()
    #read images, drop files that are not images and store it as a dataframe
    image_df = spark.read.format("image").load(sys.argv[1],dropImageFailures=True)
    #image_df.take(1)

    
    #preprocessing : getting the photo id of the images to join it to the JSON file dataframe
    split_col = pyspark.sql.functions.split(image_df['image.origin'], '/')
    image_df = image_df.withColumn('filename1', split_col.getItem(4))
    split_col1 = pyspark.sql.functions.split(image_df['filename1'], '\\.')
    image_df = image_df.withColumn('filename2', split_col1.getItem(0))
    #image_df.take(1)


    #reading the json file containing label info and joining it to image data
    path = sys.argv[2]
    photo_df = spark.read.json(path)
    final_df = image_df.join(photo_df, image_df.filename2 == photo_df.photo_id).select([col('image'),col('label')])
    #final_df.take(1)


    #mapping the string label to numeric values
    final_df= final_df.withColumn("label1", when(col("label")=='food', 1).when(col("label")=='menu', 2).when(col("label")=='drink', 3).when(col("label")=='inside', 4).otherwise(5))
    final_df = final_df.selectExpr("image as image", "label1 as label")


    #splitting the image dataset to train and test data
    final_train, final_test = final_df.randomSplit([0.8, 0.2])
    #final_train.show()


    #applying transfer learning using InceptionV3 model
    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])


    #ParamGridBuilder and cross validation for parameter tuning
    #paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.5, 0.1]).addGrid(lr.regParam, [0.1, 0.01]).build()
    #crossval = CrossValidator(estimator=p,estimatorParamMaps=paramGrid,evaluator=MulticlassClassificationEvaluator(),numFolds=4)


    #fitting the training data and transforming the test data                    
    #cvModel = crossval.fit(final_train)
    #predictions = cvModel.transform(final_test)
    yelp_model = p.fit(final_train)
    predictions = yelp_model.transform(final_test)
    #predictions.select("label1", "prediction").take(1)


    #selecting the prediction and label columns and calculating classification metrics
    predictionAndLabels = predictions.selectExpr("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator2 = MulticlassClassificationEvaluator(metricName="weightedPrecision")
    evaluator3 = MulticlassClassificationEvaluator(metricName="weightedRecall")
    evaluator4 = MulticlassClassificationEvaluator(metricName="f1")


    #printing the summary statistics
    print("Summary Stats")
    print("Accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    print("Precision = "+ str(evaluator2.evaluate(predictionAndLabels)))
    print("Recall = "+ str(evaluator3.evaluate(predictionAndLabels)))
    print("F1 Score = " + str(evaluator4.evaluate(predictionAndLabels)))
    spark.stop()

if __name__ == "__main__":
    main()






