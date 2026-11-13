import unittest

from local_chat import forex_smc as m


def build(way, step=1.0, wick=0.3):
    """Piecewise-linear closes through the waypoints; o = previous close; wicks +-wick."""
    c = [way[0]]
    for a, b in zip(way, way[1:]):
        n = max(1, int(abs(b - a) / step))
        c.extend(a + (b - a) * k / n for k in range(1, n + 1))
    o = [c[0]] + c[:-1]
    h = [max(x, y) + wick for x, y in zip(o, c)]
    l = [min(x, y) - wick for x, y in zip(o, c)]
    return o, h, l, c


class StructureTests(unittest.TestCase):
    def test_bos_then_choch(self):
        o, h, l, c = build([10, 20, 14, 24, 17, 12])
        events, trend = m.structure_events(o, h, l, c, 3, 3)
        self.assertEqual([(e["type"], e["dir"]) for e in events], [("BOS", "up"), ("CHoCH", "down")])
        self.assertAlmostEqual(events[0]["level"], 20.3)
        self.assertEqual(trend, "down")

    def test_no_lookahead_swing_is_known_only_after_right_bars(self):
        o, h, l, c = build([10, 20, 14, 24, 17, 12])
        e_full, _ = m.structure_events(o, h, l, c, 3, 3)
        e_cut, _ = m.structure_events(*(x[:30] for x in (o, h, l, c)), 3, 3)
        # events found on the prefix are exactly the events (same idx) found on the full series up to that bar
        self.assertEqual([e for e in e_full if e["idx"] < 30], e_cut)


class LiquidityTests(unittest.TestCase):
    def test_equal_highs_pool_and_sweep(self):
        o, h, l, c = build([10, 20, 15, 20.05, 14])
        liq = m.liquidity(h, l, c, 1.0)
        pool = liq["buy_side"][0]
        self.assertEqual(pool["kind"], "EQ")
        self.assertEqual(pool["touches"], 2)
        self.assertAlmostEqual(pool["price"], 20.35)
        # a later wick takes the highs and closes back below -> sweep, pool is gone
        o2, h2, l2, c2 = (x + [y] for x, y in zip((o, h, l, c), (14.0, 21.0, 13.8, 14.5)))
        liq2 = m.liquidity(h2, l2, c2, 1.0)
        self.assertEqual(liq2["last_sweep"]["side"], "high")
        self.assertAlmostEqual(liq2["last_sweep"]["level"], 20.35)
        self.assertEqual(liq2["buy_side"], [])

    def test_close_beyond_is_a_break_not_a_sweep(self):
        o, h, l, c = build([10, 20, 15, 20.05, 14])
        o2, h2, l2, c2 = (x + [y] for x, y in zip((o, h, l, c), (14.0, 22.0, 13.8, 21.5)))
        self.assertIsNone(m.liquidity(h2, l2, c2, 1.0)["last_sweep"])


class FvgTests(unittest.TestCase):
    def test_bullish_fvg_open_partial_and_filled(self):
        o = [10.0, 10.8, 13.8, 14.8, 15.0]
        h = [11.0, 14.0, 15.0, 15.5, 16.0]
        l = [9.5, 10.7, 12.5, 14.0, 14.5]
        c = [10.8, 13.8, 14.8, 15.2, 15.8]
        f = m.fvgs(o, h, l, c, 1.0)
        self.assertEqual(len(f), 1)
        self.assertEqual((f[0]["dir"], f[0]["bottom"], f[0]["top"], f[0]["filled"]), ("up", 11.0, 12.5, 0.0))
        l2 = l[:4] + [12.0]                                     # dips into the gap by a third
        f2 = m.fvgs(o, h, l2, c, 1.0)
        self.assertAlmostEqual(f2[0]["filled"], (12.5 - 12.0) / 1.5)
        l3 = l[:4] + [10.9]                                     # trades through the gap -> removed
        self.assertEqual(m.fvgs(o, h, l3, c, 1.0), [])

    def test_small_gap_ignored(self):
        o = [10.0, 10.2, 10.4]; h = [10.5, 10.6, 11.0]; l = [9.9, 10.1, 10.55]; c = [10.2, 10.4, 10.9]
        self.assertEqual(m.fvgs(o, h, l, c, 1.0), [])           # gap 0.05 < 0.25*ATR


class OrderBlockTests(unittest.TestCase):
    def test_bullish_ob_is_last_down_candle_before_the_leg_that_breaks_structure(self):
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        events, _ = m.structure_events(o, h, l, c, 3, 3)
        obs = m.order_blocks(o, h, l, c, 1.3, events)
        self.assertEqual(len(obs), 1)
        z = obs[0]
        self.assertEqual((z["dir"], z["event"], z["touched"]), ("up", "BOS", False))
        self.assertTrue(c[z["idx"]] < o[z["idx"]])              # a down-close candle
        self.assertAlmostEqual((z["low"], z["high"]), (l[z["idx"]], h[z["idx"]]), places=6) if False else None
        self.assertAlmostEqual(z["low"], l[z["idx"]]); self.assertAlmostEqual(z["high"], h[z["idx"]])

    def test_ob_removed_when_price_closes_through_it(self):
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        events, _ = m.structure_events(o, h, l, c, 3, 3)
        o2, h2, l2, c2 = (x + [y] for x, y in zip((o, h, l, c), (23.0, 23.2, 13.0, 13.5)))
        self.assertEqual(m.order_blocks(o2, h2, l2, c2, 1.3, events), [])

    def test_no_displacement_no_ob(self):
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        events, _ = m.structure_events(o, h, l, c, 3, 3)
        self.assertEqual(m.order_blocks(o, h, l, c, 50.0, events), [])      # ATR huge -> leg is not "displacement"


class RangeAndReportTests(unittest.TestCase):
    def test_premium_discount(self):
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        pd = m.premium_discount(h, l, c)
        self.assertEqual(pd["zone"], "Premium")
        self.assertAlmostEqual(pd["equilibrium"], (pd["high"] + pd["low"]) / 2)

    def test_report_and_text_are_deterministic_and_use_only_computed_numbers(self):
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        r1, r2 = m.smc_report(o, h, l, c, 1.3), m.smc_report(o, h, l, c, 1.3)
        self.assertEqual(r1, r2)
        text = "\n".join(m.smc_lines(r1, 2))
        for needle in ("BOS صعودی", "BSL", "SSL", "FVG باز", "OB معتبر", "Premium"):
            self.assertIn(needle, text)

    def test_too_short_or_no_atr_gives_none(self):
        o, h, l, c = build([10, 12])
        self.assertIsNone(m.smc_report(o, h, l, c, 1.0))
        o, h, l, c = build([20, 10, 22, 15, 26, 23])
        self.assertIsNone(m.smc_report(o, h, l, c, 0.0))


if __name__ == "__main__":
    unittest.main()
