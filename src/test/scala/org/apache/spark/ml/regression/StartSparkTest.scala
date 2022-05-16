package org.apache.spark.ml.regression
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.sql.SparkSession

@Ignore
class StartSparkTest extends AnyFlatSpec with should.Matchers {

  "Spark" should "start context" in {
    val spark = SparkSession.builder
      .appName("Custom Linear Regression Application")
      .master("local[4]")
      .getOrCreate()

    Thread.sleep(60000)
  }

}