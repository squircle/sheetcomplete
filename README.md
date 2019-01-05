# SheetComplete
**Content-aware fill for spreadsheets**


Simply provide SheetComplete with a .CSV spreadsheet (via the first argument), and it will use machine learning to fill empty cells.

**Important Notes:**

1. Input .CSV files **must have headers**, as they are used to determine data-set orientation.
2. **SheetComplete only reads numerical data**, so it works a lot better if the provided data is primarily numerical.
3. **You should not rely on this software to make important decisions**, as the quality of its predictions can vary wildly between datasets and even rows.
4. As there are multiple classifiers, the RAM usage of **SheetComplete may exceed 2GB** with large datasets.

**Possible Future Enhancements:**
* Better dataset orientation detection
* Support for CSV files without headers
* Compatibility with mixed data-types
* Smarter prediction logic for rows with multiple missing cells (simple averaging? more models?)
* Additional algorithms with non-R^2 scoring
* Add more input features to the regressors (surrounding cells not in the specific row?)
* Improved property detection
* Tune regressor parameters based on properties of data-sets
