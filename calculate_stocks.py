## CRITICAL FIX — CHART TOOLTIP LABELS AND UNIT DISPLAY

This is a display fix only. Do NOT change chart types, layout, styling,
or any other part of the app outside what is described below.

---

## FIX 1 — TOOLTIP LABELS SWAPPED ON REVENUE VS NET INCOME CHART
The tooltip is currently showing the revenue value when hovering over the
net income bar, and the net income value when hovering over the revenue bar.

Fix the tooltip so each bar label correctly matches its own value:
  - Revenue bar tooltip → shows "Revenue: $X,XXX M"
  - Net Income bar tooltip → shows "Net Income: $X,XXX M"

The data keys are:
  revenue    → "Revenue"
  net_income → "Net Income"

Ensure the label and the value come from the same key on every bar.

---

## FIX 2 — UNIT DISPLAY ON ALL THREE MONETARY CHARTS
All monetary values in financials_charts are pre-stored in millions of USD.
Do NOT divide, multiply, or convert them further.

Display rules per chart:

CHART 1 — Revenue vs Net Income
  Values are in USD millions
  Format: "$391,035 M" or "$391.0 B" if you convert for readability
  Y-axis label: "USD (Millions)"

CHART 2 — Free Cash Flow vs Total Debt
  Values are in USD millions
  Format: "$108,807 M" or "$108.8 B" if you convert for readability
  Y-axis label: "USD (Millions)"

CHART 3 — Shares Outstanding vs Buybacks
  shares_outstanding_m → number of shares in millions e.g. 15,408 M shares
  buybacks_m           → USD value of buybacks in millions e.g. $94,950 M
  These are DIFFERENT units on the same chart — use dual Y-axes or
  clearly label each series so users are not misled

CHART 4 — Returns Trajectory
  Values are already percentages e.g. 162.82 means 162.82%
  Do NOT divide by 100
  Y-axis label: "%"

---

## WHAT NOT TO CHANGE
- Do not change any other chart, tab, or component
- Do not change any data fetching logic
- Do not change the chart types or colours
