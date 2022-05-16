package org.apache.spark.ml.regression

import breeze.linalg.{DenseVector, DenseMatrix}
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  lazy val data: DataFrame = LinearRegressionTest._data.withColumn("y", predictCol(col("features")))
  lazy val labels: DenseVector[Double] = LinearRegressionTest._labels
  val predictCol: UserDefinedFunction = LinearRegressionTest._predictCol

  "Model" should "predict input data" in {
    val model: LinearRegressionModel = new LinearRegressionModel(w = Vectors.dense(1.5, 0.3, -0.7))
    model.setInputCol("features")
    model.setLabelCol("y")
    model.setOutputCol("prediction")
    validateModel(model.transform(data))
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression()
    estimator.setInputCol("features")
    estimator.setLabelCol("y")
    estimator.setOutputCol("prediction")
    val model = estimator.fit(data)
    validateModel(model.transform(data))
  }

  "Estimator" should "predict correctly" in {
    val estimator = new LinearRegression()
    estimator.setInputCol("features")
    estimator.setLabelCol("y")
    estimator.setOutputCol("prediction")
    val model = estimator.fit(data)
    model.w(0) should be(1.5 +- delta)
    model.w(1) should be(0.3 +- delta)
    model.w(2) should be(-0.7 +- delta)
  }

  private def validateModel(data: DataFrame): Unit = {
    val vector = data.select("prediction").collect()

    vector.length should be(labels.length)
    for (i <- vector.indices) {
      vector.apply(i).getDouble(0) should be(labels(i) +- delta)
    }
  }

}

object LinearRegressionTest extends WithSpark {

  lazy val _w: Vector = Vectors.dense(1.5, 0.3, -0.7)

  private lazy val _features: Matrix = Matrices.dense(1000, 3, DenseMatrix.rand[Double](1000, 3).toArray)

  lazy val _labels: DenseVector[Double] = DenseVector(_features.multiply(_w).toArray)

  lazy val _data: DataFrame = {
    import sqlContext.implicits._
    val mRows = _features.rowIter.toSeq.map(_.toArray)
    mRows.map(x => Tuple1(Vectors.dense(x))).toDF("features")
  }

  val _predictCol: UserDefinedFunction = udf { features: Any =>
    val arr = features.asInstanceOf[Vector].toArray
    1.5 * arr.apply(0) + 0.3 * arr.apply(1) - 0.7 * arr.apply(2)
  }

}