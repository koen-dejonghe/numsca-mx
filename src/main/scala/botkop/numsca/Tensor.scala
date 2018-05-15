package botkop.numsca

import org.apache.mxnet.{NDArray, Shape}

class Tensor(val array: NDArray, val isBoolean: Boolean = false) {

  override def toString: String = {
    val shape = array.shape.toString()
    val data = array.toArray.mkString(",")
    s"shape=$shape data=($data)"
  }

  def data: Array[Float] = array.toArray
  def copy(): Tensor = new Tensor(array.copy())

  def shape: Shape = array.shape
  def reshape(newShape: Shape) = new Tensor(array.reshape(newShape))
  def reshape(newShape: Array[Int]) = new Tensor(array.reshape(newShape))
  def reshape(newShape: Int*) = new Tensor(array.reshape(newShape.toArray))
  def shapeLike(t: Tensor): Tensor = reshape(t.shape)
  def transpose() = new Tensor(array.T)
  def T: Tensor = transpose()
  def transpose(axes: Array[Int]): Tensor = {
    require(axes.sorted sameElements shape.toArray.indices, "invalid axes")
    val newShape = axes.map(a => shape(a))
    reshape(newShape)
  }
  def transpose(axes: Int*): Tensor = transpose(axes.toArray)

  def round = new Tensor(NDArray.round(array))
  def abs = new Tensor(NDArray.abs(array))
  def floor = new Tensor(NDArray.floor(array))

  // unary
  def unary_- : Tensor = new Tensor(-array)

  // binary operators between 2 tensors
  def +(other: Tensor) = new Tensor(array + other.array)
  def -(other: Tensor) = new Tensor(array - other.array)
  def *(other: Tensor) = new Tensor(array * other.array)
  def /(other: Tensor) = new Tensor(array / other.array)
  def %(other: Tensor) = new Tensor(array % other.array)
  def dot(other: Tensor) = new Tensor(NDArray.dot(array, other.array))
  def **(other: Tensor) = new Tensor(array ** other.array)

  def :=(other: Tensor): Unit = array set other.array
  def +=(other: Tensor): Unit = array += other.array
  def -=(other: Tensor): Unit = array -= other.array
  def *=(other: Tensor): Unit = array *= other.array
  def /=(other: Tensor): Unit = array /= other.array
  def %=(other: Tensor): Unit = array %= other.array
  def **=(other: Tensor): Unit = array **= other.array

  // binary operators between tensor and float
  def +(f: Float) = new Tensor(array + f)
  def -(f: Float) = new Tensor(array - f)
  def *(f: Float) = new Tensor(array * f)
  def /(f: Float) = new Tensor(array / f)
  def %(f: Float) = new Tensor(array % f)
  def **(f: Float) = new Tensor(array ** f)

  def :=(f: Float): Unit = array set f
  def +=(f: Float): Unit = array += f
  def -=(f: Float): Unit = array -= f
  def *=(f: Float): Unit = array *= f
  def /=(f: Float): Unit = array /= f
  def %=(f: Float): Unit = array %= f
  def **=(f: Float): Unit = array **= f

  // binary boolean operators between 2 tensors
  def &&(other: Tensor): Tensor = {
    require(this.isBoolean && other.isBoolean)
    new Tensor(this.array * other.array, true)
  }

  def ||(other: Tensor): Tensor = {
    require(this.isBoolean && other.isBoolean)
    new Tensor(NDArray.max(this.array + other.array, 1.0), true)
  }

  def ==(other: Tensor): Tensor =
    new Tensor(NDArray.equal(this.array, other.array), true)
  def !=(other: Tensor): Tensor =
    new Tensor(NDArray.notEqual(this.array, other.array), true)
  def >=(other: Tensor): Tensor =
    new Tensor(NDArray.greaterEqual(this.array, other.array), true)
  def >(other: Tensor): Tensor =
    new Tensor(NDArray.greater(this.array, other.array), true)
  def <=(other: Tensor): Tensor =
    new Tensor(NDArray.lesserEqual(this.array, other.array), true)
  def <(other: Tensor): Tensor =
    new Tensor(NDArray.lesser(this.array, other.array), true)

  // binary boolean operators between tensor and float
  def ==(f: Float): Tensor =
    new Tensor(NDArray.equal(this.array, f), true)
  def !=(f: Float): Tensor =
    new Tensor(NDArray.notEqual(this.array, f), true)
  def >=(f: Float): Tensor =
    new Tensor(NDArray.greaterEqual(this.array, f), true)
  def >(f: Float): Tensor = new Tensor(NDArray.greater(this.array, f), true)
  def <=(f: Float): Tensor =
    new Tensor(NDArray.lesserEqual(this.array, f), true)
  def <(f: Float): Tensor = new Tensor(NDArray.lesser(this.array, f), true)

  def maximum(other: Tensor) =
    new Tensor(NDArray.maximum(array, other.array))
  def minimum(other: Tensor) =
    new Tensor(NDArray.minimum(array, other.array))
  def maximum(f: Float) = new Tensor(NDArray.maximum(array, f))
  def minimum(f: Float) = new Tensor(NDArray.minimum(array, f))

  def slice(i: Int) = new Tensor(array.slice(i))
  def slice(i: Int, dim: Int): Tensor = {
    val m = Map("axis" -> dim, "begin" -> i, "end" -> (i + 1))
    new Tensor(NDArray.slice_axis(m)(array))
  }

  def squeeze(): Float = array.toScalar

  // todo: there is probably a better way
  def apply(index: Int*) =
    new Tensor(index.zipWithIndex.foldLeft(array) {
      case (arr, (ix, si)) => arr.at(handleNegIndex(ix, si))
    })

  private def handleNegIndex(i: Int, shapeIndex: Int) =
    if (i < 0) shape(shapeIndex) + i else i
}

object Tensor {
  def apply(data: Array[Float], shape: Shape): Tensor = {
    val array = NDArray.array(data, shape)
    new Tensor(array)
  }

  def apply(data: Float*): Tensor = {
    val array = NDArray.array(data.toArray, Shape(1, data.length))
    new Tensor(array)
  }

}
