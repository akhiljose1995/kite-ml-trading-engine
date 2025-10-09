#!/bin/bash
cd "D:/My Automation Works/trading_automation/trading_automation/new/kite-backtester"
python.exe ./tests/oi_tracker_test.py nifty50 >> startup_log.txt 2>&1
read -p "Press any key to close..."