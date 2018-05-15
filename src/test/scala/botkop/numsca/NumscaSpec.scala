package botkop.numsca

import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

class NumscaSpec extends FlatSpec with Matchers {

  val ta: Tensor = ns.arange(10)
  val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
  val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)

  "A Tensor" should "transpose over multiple dimensions" in {
    val x = ns.arange(6).reshape(1, 2, 3)
    val y = ns.transpose(x, 1, 0, 2)
    val z = ns.reshape(x, 2, 1, 3)
    assert(ns.arrayEqual(y, z))
  }

  it should "retrieve the correct elements" in {
    // todo: implicitly convert tensor to double when only 1 element?
    assert(ta(1).squeeze() == 1)
    assert(tb(1, 0).squeeze() == 3)
    assert(tc(1, 0, 2).squeeze() == 14)

    val i = List(1, 0, 1)
    assert(tc(i: _*).squeeze() == 13)
  }

  it should "change array values in place" in {
    val t = ta.copy()
    t(3) := -5
    assert(t.data sameElements Array(0, 1, 2, -5, 4, 5, 6, 7, 8, 9))
    t(0) += 7
    assert(t.data sameElements Array(7, 1, 2, -5, 4, 5, 6, 7, 8, 9))

    val t2 = tb.copy()
    t2(2, 1) := -7
    t2(1, 2) := -3
    assert(
      arrayEqual(t2,
        Tensor(0.00f, 1.00f, 2.00f, 3.00f, 4.00f, -3.00f, 6.00f, -7.00f,
          8.00f).reshape(3, 3)))
  }

}
