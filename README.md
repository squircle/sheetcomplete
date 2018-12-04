# sheetcomplete
**Content-aware fill for spreadsheets**


Simply provide sheetcomplete with a .CSV spreadsheet (via the first argument), and it will use machine learning to fill empty cells.

**Important Notes:**

1. Input .CSV files **must have headers**, as they are used to determine data-set orientation.
2. **sheetcomplete only reads numerical data**, so it works a lot better if the provided data is primarily numerical.
3. **You should not rely on this software to make important decisions**, as the quality of its predictions can vary wildly between datasets and even rows.
