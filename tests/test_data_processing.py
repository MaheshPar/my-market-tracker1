"""
tests/test_data_processing.py - Comprehensive Unit Tests for Financial & Technical Engines
Run via: python -m unittest discover tests
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.data_processing import (
    calculate_xirr,
    get_xirr_weighted_buy_date,
    process_lofo,
    split_lots_into_blocks,
    calculate_wilder_rsi,
    parse_tradebook,
    compute_score,
    score_label,
    detect_swing_pivots,
    is_market_open,
    apply_live_prices
)


class TestCalculateXirr(unittest.TestCase):
    def test_simple_doubling_in_one_year(self):
        # Invest 10,000 on 2023-01-01, receive 20,000 on 2024-01-01
        d1 = datetime(2023, 1, 1)
        d2 = datetime(2024, 1, 1)
        xirr = calculate_xirr([-10000.0, 20000.0], [d1, d2])
        # 100% return in 1 year (with leap year offset ~365.25 days it's around 100% +- 1%)
        self.assertAlmostEqual(xirr, 100.0, delta=1.5)

    def test_loss_scenario(self):
        # Invest 10,000 on 2023-01-01, receive 5,000 on 2024-01-01 (-50% return)
        d1 = datetime(2023, 1, 1)
        d2 = datetime(2024, 1, 1)
        xirr = calculate_xirr([-10000.0, 5000.0], [d1, d2])
        self.assertAlmostEqual(xirr, -50.0, delta=2.0)

    def test_multi_leg_cashflows(self):
        d1 = datetime(2023, 1, 1)
        d2 = datetime(2023, 7, 1)
        d3 = datetime(2024, 1, 1)
        xirr = calculate_xirr([-5000.0, -5000.0, 11000.0], [d1, d2, d3])
        self.assertTrue(xirr > 0.0)

    def test_all_positive_or_all_negative(self):
        d1 = datetime(2023, 1, 1)
        d2 = datetime(2024, 1, 1)
        # All positive -> invalid, should return 0
        self.assertEqual(calculate_xirr([1000.0, 2000.0], [d1, d2]), 0.0)
        # All negative -> invalid, should return 0
        self.assertEqual(calculate_xirr([-1000.0, -2000.0], [d1, d2]), 0.0)

    def test_empty_or_mismatched_inputs(self):
        d1 = datetime(2023, 1, 1)
        self.assertEqual(calculate_xirr([], []), 0.0)
        self.assertEqual(calculate_xirr([-100.0], [d1]), 0.0)
        self.assertEqual(calculate_xirr([-100.0, 200.0], [d1]), 0.0)

    def test_same_day_cashflows(self):
        d1 = datetime(2023, 1, 1)
        self.assertEqual(calculate_xirr([-100.0, 100.0], [d1, d1]), 0.0)


class TestGetXirrWeightedBuyDate(unittest.TestCase):
    def test_single_lot(self):
        lot_date = pd.Timestamp(2023, 5, 1)
        lots = [{'price': 100.0, 'date': lot_date}]
        target_date = pd.Timestamp(2024, 5, 1)
        w_date = get_xirr_weighted_buy_date(lots, 150.0, 50.0, target_date)
        # Should be approximately 1 year before target_date
        diff_days = abs((w_date - lot_date).days)
        self.assertTrue(diff_days <= 5)

    def test_zero_or_negative_valuation(self):
        lot_date = pd.Timestamp(2023, 5, 1)
        lots = [{'price': 100.0, 'date': lot_date}]
        target_date = pd.Timestamp(2024, 5, 1)
        self.assertEqual(get_xirr_weighted_buy_date(lots, 0.0, 10.0, target_date), lot_date)
        self.assertEqual(get_xirr_weighted_buy_date(lots, -50.0, 10.0, target_date), lot_date)

    def test_empty_lots(self):
        target_date = pd.Timestamp(2024, 5, 1)
        self.assertEqual(get_xirr_weighted_buy_date([], 100.0, 10.0, target_date), target_date)


class TestProcessLofo(unittest.TestCase):
    def test_lofo_lowest_price_sold_first(self):
        # Buy 10 @ 100 on Jan 1
        # Buy 10 @ 80 on Jan 2
        # Sell 10 @ 120 on Jan 3
        # Under LOFO, the 80 price lot must be sold first!
        # Realized profit = 10 * (120 - 80) = 400
        # Remaining lot must be 10 @ 100 = 1000
        trades = pd.DataFrame([
            {
                'Symbol': 'TEST',
                'Exchange': 'NSE',
                'Trade Type': 'buy',
                'Trade Date': pd.Timestamp('2023-01-01'),
                'Order Execution Time': pd.Timestamp('2023-01-01 10:00'),
                'Quantity': 10,
                'Price': 100.0
            },
            {
                'Symbol': 'TEST',
                'Exchange': 'NSE',
                'Trade Type': 'buy',
                'Trade Date': pd.Timestamp('2023-01-02'),
                'Order Execution Time': pd.Timestamp('2023-01-02 10:00'),
                'Quantity': 10,
                'Price': 80.0
            },
            {
                'Symbol': 'TEST',
                'Exchange': 'NSE',
                'Trade Type': 'sell',
                'Trade Date': pd.Timestamp('2023-01-03'),
                'Order Execution Time': pd.Timestamp('2023-01-03 10:00'),
                'Quantity': 10,
                'Price': 120.0
            }
        ])

        closed, open_data = process_lofo(trades)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]['Buy Price'], 80.0)
        self.assertEqual(closed[0]['Sell Price'], 120.0)
        self.assertEqual(closed[0]['Qty'], 10)
        self.assertEqual(closed[0]['Profit'], 400.0)

        # Open lot check
        self.assertIn('TEST', open_data)
        self.assertEqual(open_data['TEST']['remaining_qty'], 10)
        self.assertEqual(open_data['TEST']['invested_amount'], 1000.0)
        self.assertEqual(open_data['TEST']['avg_price'], 100.0)
        self.assertEqual(open_data['TEST']['realized_profit'], 400.0)

    def test_partial_sell(self):
        trades = pd.DataFrame([
            {
                'Symbol': 'XYZ',
                'Exchange': 'NSE',
                'Trade Type': 'buy',
                'Trade Date': pd.Timestamp('2023-01-01'),
                'Order Execution Time': pd.Timestamp('2023-01-01 10:00'),
                'Quantity': 20,
                'Price': 50.0
            },
            {
                'Symbol': 'XYZ',
                'Exchange': 'NSE',
                'Trade Type': 'sell',
                'Trade Date': pd.Timestamp('2023-01-05'),
                'Order Execution Time': pd.Timestamp('2023-01-05 11:00'),
                'Quantity': 8,
                'Price': 60.0
            }
        ])
        closed, open_data = process_lofo(trades)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]['Qty'], 8)
        self.assertEqual(closed[0]['Profit'], 80.0)

        self.assertEqual(open_data['XYZ']['remaining_qty'], 12)
        self.assertEqual(open_data['XYZ']['invested_amount'], 12 * 50.0)

    def test_same_day_sells_clubbing(self):
        # Two sells on the same day should be clubbed at weighted avg price
        # Buy 30 @ 100
        # Sell 10 @ 110 at 10:00, Sell 10 @ 120 at 12:00 -> Avg sell price = 115
        trades = pd.DataFrame([
            {
                'Symbol': 'CLUB',
                'Exchange': 'NSE',
                'Trade Type': 'buy',
                'Trade Date': pd.Timestamp('2023-02-01'),
                'Order Execution Time': pd.Timestamp('2023-02-01 09:30'),
                'Quantity': 30,
                'Price': 100.0
            },
            {
                'Symbol': 'CLUB',
                'Exchange': 'NSE',
                'Trade Type': 'sell',
                'Trade Date': pd.Timestamp('2023-02-02'),
                'Order Execution Time': pd.Timestamp('2023-02-02 10:00'),
                'Quantity': 10,
                'Price': 110.0
            },
            {
                'Symbol': 'CLUB',
                'Exchange': 'NSE',
                'Trade Type': 'sell',
                'Trade Date': pd.Timestamp('2023-02-02'),
                'Order Execution Time': pd.Timestamp('2023-02-02 12:00'),
                'Quantity': 10,
                'Price': 120.0
            }
        ])
        closed, open_data = process_lofo(trades)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]['Qty'], 20)
        self.assertEqual(closed[0]['Sell Price'], 115.0)
        self.assertEqual(closed[0]['Profit'], 300.0)
        self.assertEqual(open_data['CLUB']['remaining_qty'], 10)

    def test_empty_dataframe(self):
        closed, open_data = process_lofo(pd.DataFrame())
        self.assertEqual(closed, [])
        self.assertEqual(open_data, {})


class TestSplitLotsIntoBlocks(unittest.TestCase):
    def test_block_splitting_with_partial(self):
        # Open data with lots: 50 shares @ 100 = 5000, 50 shares @ 100 = 5000 -> Total 10,000
        # Block size = 8000 -> Block 1: 80 shares (8000), Block 2 (P): 20 shares (2000)
        lots = [
            {'date': pd.Timestamp('2023-01-01'), 'order_time': pd.Timestamp('2023-01-01 10:00'),
             'quantity': 50, 'price': 100.0, 'remaining_qty': 50},
            {'date': pd.Timestamp('2023-01-02'), 'order_time': pd.Timestamp('2023-01-02 10:00'),
             'quantity': 50, 'price': 100.0, 'remaining_qty': 50}
        ]
        open_data = {
            'TEST': {
                'symbol': 'TEST',
                'remaining_qty': 100,
                'lots': lots
            }
        }
        blocks = split_lots_into_blocks(open_data, block_size=8000)
        self.assertIn('TEST', blocks)
        sym_blks = blocks['TEST']
        self.assertEqual(len(sym_blks), 2)
        self.assertEqual(sym_blks[0]['block_name'], 'Block 1')
        self.assertEqual(sym_blks[0]['qty'], 80)
        self.assertEqual(sym_blks[1]['block_name'], 'Block 2 (P)')
        self.assertEqual(sym_blks[1]['qty'], 20)

    def test_liquidcase_special_handling(self):
        lots = [
            {'date': pd.Timestamp('2023-01-01'), 'order_time': pd.Timestamp('2023-01-01 10:00'),
             'quantity': 100, 'price': 100.0, 'remaining_qty': 100}
        ]
        open_data = {
            'LIQUIDCASE': {
                'symbol': 'LIQUIDCASE',
                'remaining_qty': 100,
                'lots': lots
            }
        }
        blocks = split_lots_into_blocks(open_data, block_size=1000)
        self.assertIn('LIQUIDCASE', blocks)
        self.assertEqual(len(blocks['LIQUIDCASE']), 1)
        self.assertEqual(blocks['LIQUIDCASE'][0]['block_name'], 'Block 1')


class TestCalculateWilderRsi(unittest.TestCase):
    def test_strongly_increasing_prices(self):
        prices = pd.Series([10.0 + i for i in range(30)])
        rsi = calculate_wilder_rsi(prices, period=14)
        self.assertTrue(rsi.iloc[-1] > 90.0)

    def test_strongly_decreasing_prices(self):
        prices = pd.Series([100.0 - i for i in range(30)])
        rsi = calculate_wilder_rsi(prices, period=14)
        self.assertTrue(rsi.iloc[-1] < 10.0)

    def test_empty_series(self):
        rsi = calculate_wilder_rsi(pd.Series(dtype=float))
        self.assertTrue(rsi.empty)


class TestParseTradebook(unittest.TestCase):
    def test_parse_valid_tradebook(self):
        df_raw = pd.DataFrame({
            ' Symbol ': [' reliance ', ' TCS '],
            ' Exchange ': [' nse ', ' bse '],
            'Trade Type': ['BUY', 'SELL'],
            'Quantity': ['10', '5'],
            'Price': ['2500.50', '3400.00'],
            'Trade Date': ['2023-05-10', '2023-05-12'],
            'Order Execution Time': ['2023-05-10 10:30', '2023-05-12 11:00']
        })
        parsed = parse_tradebook(df_raw)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed['Symbol'].tolist(), ['RELIANCE', 'TCS'])
        self.assertEqual(parsed['Exchange'].tolist(), ['NSE', 'BSE'])
        self.assertEqual(parsed['Trade Type'].tolist(), ['buy', 'sell'])
        self.assertEqual(parsed['Quantity'].tolist(), [10, 5])
        self.assertEqual(parsed['Price'].tolist(), [2500.5, 3400.0])

    def test_filters_invalid_and_zero_quantity(self):
        df_raw = pd.DataFrame({
            'Symbol': ['A', 'B', 'C'],
            'Trade Type': ['buy', 'invalid', 'sell'],
            'Quantity': [10, 5, 0],
            'Price': [100.0, 100.0, 100.0]
        })
        parsed = parse_tradebook(df_raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed['Symbol'].iloc[0], 'A')


class TestComputeScoreAndLabels(unittest.TestCase):
    def test_high_score_oversold(self):
        metrics = {
            "LTP": 80.0,
            "RSI": 32.0,            # +1
            "200_DMA": 100.0,        # +1 (LTP < 200_DMA)
            "BB_LOW": 85.0,          # +1 (LTP < BB_LOW)
            "vs_52W_Pct": -25.0,     # +2 (<= -20%)
            "ATH_Dist_Pct": -30.0    # +1 (<= -20%)
        }
        score, reasons = compute_score(metrics)
        self.assertEqual(score, 6)
        self.assertIn("RSI<40", reasons)
        self.assertIn("<200DMA", reasons)
        self.assertIn("<BBLow", reasons)
        self.assertEqual(score_label(score), "🔴 Strong")

    def test_low_score_bullish(self):
        metrics = {
            "LTP": 120.0,
            "RSI": 65.0,
            "200_DMA": 100.0,
            "BB_LOW": 95.0,
            "vs_52W_Pct": -2.0,
            "ATH_Dist_Pct": -2.0
        }
        score, reasons = compute_score(metrics)
        self.assertEqual(score, 0)
        self.assertEqual(score_label(score), "⚪ Watch")


class TestMarketHours(unittest.TestCase):
    def test_weekend_closed(self):
        # Saturday: 2026-11-14
        sat = datetime(2026, 11, 14, 11, 0)
        self.assertFalse(is_market_open(sat))
        # Sunday: 2026-11-15
        sun = datetime(2026, 11, 15, 11, 0)
        self.assertFalse(is_market_open(sun))

    def test_weekday_open_hours(self):
        # Wednesday 11:30 AM
        wed_open = datetime(2026, 11, 11, 11, 30)
        self.assertTrue(is_market_open(wed_open))

    def test_weekday_closed_hours(self):
        # Wednesday 08:00 AM (Pre-market)
        wed_early = datetime(2026, 11, 11, 8, 0)
        self.assertFalse(is_market_open(wed_early))
        # Wednesday 04:00 PM (Post-market)
        wed_late = datetime(2026, 11, 11, 16, 0)
        self.assertFalse(is_market_open(wed_late))


class TestDetectSwingPivots(unittest.TestCase):
    def test_zigzag_pivots(self):
        # Generate artificial swing pattern
        dates = pd.date_range("2023-01-01", periods=100, freq="B")
        # 100 -> 80 (-20%) -> 110 (+37%) -> 90 (-18%) -> 120 (+33%)
        prices = [100.0]
        for i in range(1, 25):
            prices.append(prices[-1] - 0.8)   # Down to ~80
        for i in range(25, 50):
            prices.append(prices[-1] + 1.2)   # Up to ~110
        for i in range(50, 75):
            prices.append(prices[-1] - 0.8)   # Down to ~90
        for i in range(75, 100):
            prices.append(prices[-1] + 1.2)   # Up to ~120

        df = pd.DataFrame({"Close": prices}, index=dates)
        res = detect_swing_pivots(df, threshold_pct=7.0)
        self.assertIsNotNone(res)
        self.assertTrue(res["pivot_count"] >= 3)
        self.assertIn(res["status"], ["PEAK", "FLOOR"])


class TestApplyLivePrices(unittest.TestCase):
    def test_apply_live_prices(self):
        feeds = {
            'TEST': {
                'LTP': 100.0,
                'ATH': 150.0,
                '52W_High': 125.0,
                'Day_Change_Pct': 0.0,
                'ATH_Dist_Pct': -33.3,
                'vs_52W_Pct': -20.0
            }
        }
        live = {
            'TEST': {
                'ltp': 120.0,
                'day_chg_pct': 2.5
            }
        }
        apply_live_prices(feeds, live)
        self.assertEqual(feeds['TEST']['LTP'], 120.0)
        self.assertEqual(feeds['TEST']['Day_Change_Pct'], 2.5)
        self.assertAlmostEqual(feeds['TEST']['ATH_Dist_Pct'], (120 - 150) / 150 * 100.0)
        self.assertAlmostEqual(feeds['TEST']['vs_52W_Pct'], (120 - 125) / 125 * 100.0)


if __name__ == '__main__':
    unittest.main()

