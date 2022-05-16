package org.apache.spark.ml.regression

import breeze.linalg
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol, HasTol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.mllib.linalg.{Vectors => MLLibVectors}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.{ml, mllib}

trait LinearRegressionParams extends PredictorParams with HasInputCol with HasLabelCol with HasOutputCol{
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String) : this.type = set(outputCol, value)
  def setLabelCol(value: String) : this.type = set(labelCol, value)
  def setGradRate(value: Double): this.type = set(gradRate, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  final val gradRate: Param[Double] = new DoubleParam(
    this,
    "gradientRate",
    "Gradient descent rate"
  )
  def getGradRate: Double = $(gradRate)
  val maxIter = new IntParam(
    this, "maxIter", "maximum iterations"
  )
  def getMaxIter: Int = $(maxIter)

  setDefault(maxIter -> 1500, gradRate -> 4e-2)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()
    val asm = new VectorAssembler().setInputCols(Array(getInputCol, getLabelCol)).setOutputCol("result")
    val vectors = asm.transform(dataset).select("result").as[Vector]

    val cnt = vectors.first().size - 1
    val maxIter = getMaxIter
    val gRate = getGradRate
    val w = linalg.Vector.zeros[Double](cnt)
    for (i <- 1 to maxIter) {
      val grads = vectors.rdd.mapPartitions(iter => {
        val summarizer = new MultivariateOnlineSummarizer()
        iter.foreach(item => {
          val row: linalg.DenseVector[Double] = item.asBreeze.toDenseVector
          val y = row(-1)
          val x = row(0 until  cnt).toDenseVector
          summarizer.add(MLLibVectors.fromBreeze( (x.dot(w) - y) * x)  )
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      w -= 2 * gRate * grads.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(mllib.linalg.Vectors.fromBreeze(w).asML).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

class LinearRegressionModel(override val uid: String,
                            val w: Vector,
                                 ) extends Model[LinearRegressionModel] with LinearRegressionParams {

  private[regression] def this(w: Vector) = this(
    Identifiable.randomUID("linearRegressionModel"),
    w.toDense
  )

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = false)
    val predictUDF = udf { features: Any =>{
      predict(features.asInstanceOf[Vector])
    }
    }
    dataset.withColumn($(outputCol), predictUDF(dataset($(inputCol))), outputSchema($(outputCol)).metadata)
  }

private def predict(features: Vector) =  features.asBreeze.dot(w.asBreeze)
  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(w), extra)

  def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val tupled = Tuple1(w)
      sqlContext.createDataFrame(Seq(tupled)).write.parquet(path + "/vectors")
    }
  }

}
object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/vectors")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()
      val weight = vectors.select(vectors("_1").as[Vector]).first().asBreeze.toDenseVector
      val model = new LinearRegressionModel(mllib.linalg.Vectors.fromBreeze(weight).asML)
      metadata.getAndSetParams(model)
      model
    }
  }
}