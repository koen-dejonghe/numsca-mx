package botkop

import org.apache.mxnet.Shape
import org.apache.mxnet.NDArray

import scala.language.implicitConversions

package object numsca {

  implicit class MxNumscaFloatOps(d: Float) {
    def +(t: Tensor): Tensor = t + d
    def -(t: Tensor): Tensor = -t + d
    def *(t: Tensor): Tensor = t * d
    def /(t: Tensor): Tensor = (t ** -1) * d
  }

  case class MxNumscaRange(from: Int, to: Option[Int])

  def :>(end: Int) = MxNumscaRange(0, Some(end))
  def :> = MxNumscaRange(0, None)

  implicit class MxNumscaInt(i: Int) {
    def :>(end: Int) = MxNumscaRange(i, Some(end))
    def :> = MxNumscaRange(i, None)
  }

  implicit def intToMxNumscaRange(i: Int): MxNumscaRange =
    MxNumscaRange(i, Some(i + 1))

  def array(ds: Float*) = Tensor(ds: _*)

  def zeros(shape: Shape) = new Tensor(NDArray.zeros(shape))
  def zeros(shape: Int*) = new Tensor(NDArray.zeros(shape: _*))
  def zerosLike(t: Tensor) = new Tensor(NDArray.zeros(t.shape))

  def ones(shape: Shape) = new Tensor(NDArray.ones(shape))
  def ones(shape: Int*) = new Tensor(NDArray.ones(shape: _*))
  def onesLike(t: Tensor) = new Tensor(NDArray.ones(t.shape))

  def full(shape: Shape, f: Float) = new Tensor(NDArray.full(shape, f))

  def arange(end: Float): Tensor = new Tensor(NDArray.arange(end))
  def arange(start: Float, end: Float): Tensor =
    new Tensor(NDArray.arange(start, Some(end)))

  def randn(shape: Shape): Tensor =
    new Tensor(NDArray.random_normal(Map("shape" -> shape))())
  def randn(shape: Int*): Tensor = randn(Shape(shape))

  def rand(shape: Shape): Tensor =
    new Tensor(NDArray.random_uniform(Map("shape" -> shape))())
  def rand(shape: Int*): Tensor = rand(Shape(shape))

  def randint(low: Int, shape: Shape): Tensor = {
    val data = Array.fill(shape.product)(scala.util.Random.nextInt(low).toFloat)
    new Tensor(NDArray.array(data, shape))
  }

  def uniform(low: Float = 0f, high: Float = 1f, shape: Shape): Tensor =
    new Tensor(
      NDArray.random_uniform(
        Map("low" -> low, "high" -> high, "shape" -> shape))())

  def linspace(start: Float, stop: Float, num: Int): Tensor = {
    val increment = (stop - start) / (num - 1)
    val data = Array.tabulate(num)(i => start + i * increment)
    new Tensor(NDArray.array(data, Shape(num)))
  }

  def copy(t: Tensor): Tensor = t.copy()

  def abs(t: Tensor): Tensor = t.abs
  def floor(t: Tensor): Tensor = t.floor
  def round(t: Tensor): Tensor = t.round

  def maximum(t: Tensor, f: Float): Tensor = t.maximum(f)
  def maximum(a: Tensor, b: Tensor): Tensor = a.maximum(b)
  def minimum(t: Tensor, f: Float): Tensor = t.minimum(f)
  def minimum(a: Tensor, b: Tensor): Tensor = a.minimum(b)

  def sum(t: Tensor): Float = NDArray.sum(t.array).toScalar
  def sum(t: Tensor, axis: Int*): Tensor = {
    val arr = NDArray.sum(Map("data" -> t.array, "axis" -> axis))()
    new Tensor(arr)
  }

  def prod(t: Tensor): Float = NDArray.prod(t.array).toScalar
  def prod(t: Tensor, axis: Int*): Tensor = {
    val arr = NDArray.prod(Map("data" -> t.array, "axis" -> axis))()
    new Tensor(arr)
  }

  def reshape(x: Tensor, shape: Shape): Tensor = x.reshape(shape)
  def reshape(x: Tensor, shape: Array[Int]): Tensor = x.reshape(shape)
  def reshape(x: Tensor, shape: Int*): Tensor = x.reshape(shape: _*)

  def transpose(x: Tensor): Tensor = x.transpose()
  def transpose(x: Tensor, axes: Int*): Tensor = x.transpose(axes: _*)
  def transpose(x: Tensor, axes: Array[Int]): Tensor = x.transpose(axes)

  def arrayEqual(t1: Tensor, t2: Tensor): Boolean = numsca.prod(t1 == t2) == 1
}
