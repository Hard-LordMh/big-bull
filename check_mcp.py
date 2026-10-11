"""
check_mcp.py
------------
Diagnostic script to test and verify the MCP Server handshake, tools, and resources.
"""

import sys
import asyncio
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from mcp_server import mcp


async def verify_mcp():
    print("=" * 60)
    print(">> BIG BULL AI - MODEL CONTEXT PROTOCOL (MCP) HEALTH CHECK")
    print("=" * 60)

    # 1. Tools check
    tools = await mcp.list_tools()
    print(f"\n[1] Tools Registered: {len(tools)}")
    for t in tools:
        print(f"    ✓ {t.name}")

    # 2. Resources check
    resources = await mcp.list_resources()
    print(f"\n[2] Resources Registered: {len(resources)}")
    for r in resources:
        print(f"    ✓ {r.uri}")

    # 3. Prompts check
    prompts = await mcp.list_prompts()
    print(f"\n[3] Prompts Registered: {len(prompts)}")
    for p in prompts:
        print(f"    ✓ {p.name}")

    # 4. Live Tool Invocation Check
    print("\n[4] Live Tool Execution Test: 'predict_stock_direction'")
    call_result = await mcp.call_tool("predict_stock_direction", {"ticker": "RELIANCE.NS", "horizon": 5})
    
    # In MCP 2.x, CallToolResult holds a list of Content objects in call_result.content
    raw_text = call_result.content[0].text
    data = json.loads(raw_text)

    print(f"    * Ticker: {data.get('ticker')}")
    print(f"    * Predicted Direction: {data.get('predicted_direction')}")
    print(f"    * Probability UP: {data.get('probability_up_pct')}%")
    print(f"    * Probability DOWN: {data.get('probability_down_pct')}%")
    print(f"    * Holdout Test Accuracy: {data.get('test_accuracy_pct')}%")
    print(f"    * Outperforms Baseline: {data.get('outperforms_baseline')}")

    # 5. Live Tool Invocation Check for Timing
    print("\n[5] Live Tool Execution Test: 'get_buy_sell_timing'")
    timing_res = await mcp.call_tool("get_buy_sell_timing", {"ticker": "TCS.NS", "horizon": 5})
    timing_data = json.loads(timing_res.content[0].text)
    print(f"    * Ticker: TCS.NS")
    print(f"    * Timing Signal: {timing_data.get('signal')}")
    print(f"    * Reference Price: Rs {timing_data.get('reference_price')}")

    # 6. Live Tool Invocation Check for Random Forest Detection
    print("\n[6] Live Tool Execution Test: 'detect_market_regime_and_anomalies'")
    rf_res = await mcp.call_tool("detect_market_regime_and_anomalies", {"ticker": "RELIANCE.NS"})
    rf_data = json.loads(rf_res.content[0].text)
    tc = rf_data.get("tree_consensus", {})
    mr = rf_data.get("market_regime", {})
    ba = rf_data.get("breakout_anomaly", {})
    print(f"    * Random Forest Tree Consensus: {tc.get('trees_up')}/{tc.get('total_trees')} Trees UP ({tc.get('consensus_pct')}%) [{tc.get('strength_label')}]")
    print(f"    * Market Regime: {mr.get('badge')} ({mr.get('regime_name')})")
    print(f"    * Isolation Forest Anomaly: {ba.get('badge')}")

    print("\n" + "=" * 60)
    print(">> STATUS: MCP SERVER IS 100% HEALTHY, CONNECTED, AND OPERATIONAL!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_mcp())
