import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from standard_quant_tools.agent.models import BacktestInput, AnalysisInput
from standard_quant_tools.agent.tools import run_sma_backtest, analyze_stock_risk

def test_agent_tools():
    print("Testing Agent Tools...")
    
    # Test 1: Analysis Tool
    print("\n1. Testing 'analyze_stock_risk' for NVDA vs SPY...")
    try:
        analysis_input = AnalysisInput(symbol="NVDA", benchmark="SPY", period="1y")
        result = analyze_stock_risk(analysis_input)
        print("Success! Result:")
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Error in analysis tool: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Backtest Tool
    print("\n2. Testing 'run_sma_backtest' for AAPL...")
    try:
        backtest_input = BacktestInput(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2024-01-01",
            strategy_type="sma_crossover",
            parameters={"fast_period": 10, "slow_period": 30},
            initial_capital=10000.0
        )
        result = run_sma_backtest(backtest_input)
        print("Success! Result Summary:")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Final Equity: ${result.final_equity:,.2f}")
    except Exception as e:
        print(f"Error in backtest tool: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_tools()
