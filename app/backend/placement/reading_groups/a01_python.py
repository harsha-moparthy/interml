from . import G

PD = "https://pandas.pydata.org/docs"
VP = "https://jakevdp.github.io/PythonDataScienceHandbook"
NP = "https://numpy.org/doc/stable"

DSINTRO = ("Intro to data structures", f"{PD}/user_guide/dsintro.html", "pandas docs", "Official guide to building a Series or DataFrame from dicts, lists and arrays.")
SUBSET = ("How do I select a subset of a DataFrame?", f"{PD}/getting_started/intro_tutorials/03_subset_data.html", "pandas docs", "Step-by-step tutorial on selecting columns, rows and boolean-filtered subsets.")
INDEXING = ("Indexing and selecting data", f"{PD}/user_guide/indexing.html", "pandas docs", "Reference for loc, iloc, boolean indexing and set_index/reset_index.")
HB_INDEX = ("Data Indexing and Selection (Python Data Science Handbook)", f"{VP}/03.02-data-indexing-and-selection.html", "Jake VanderPlas", "Explains loc and iloc and why both exist, with small runnable examples.")
HB_MISSING = ("Handling Missing Data (Python Data Science Handbook)", f"{VP}/03.04-missing-values.html", "Jake VanderPlas", "Compares NaN and None, and shows isnull, dropna and fillna.")
GROUPBY = ("Group by: split-apply-combine", f"{PD}/user_guide/groupby.html", "pandas docs", "Official guide to aggregation, transformation and filtration on groups.")
RP_GROUPBY = ("Pandas GroupBy: Your Guide to Grouping Data in Python", "https://realpython.com/pandas-groupby/", "Real Python", "Long tutorial that builds up the split-apply-combine model with real data.")
RESHAPE = ("Reshaping and pivot tables", f"{PD}/user_guide/reshaping.html", "pandas docs", "Official guide to pivot, melt, stack/unstack and crosstab.")
MERGING = ("Merge, join, concatenate and compare", f"{PD}/user_guide/merging.html", "pandas docs", "Official guide to concat and merge with diagrams of each join type.")
BROADCAST = ("Broadcasting", f"{NP}/user/basics.broadcasting.html", "NumPy docs", "States the broadcasting rule and shows shape-compatibility examples.")
HB_BROADCAST = ("Computation on Arrays: Broadcasting (Python Data Science Handbook)", f"{VP}/02.05-computation-on-arrays-broadcasting.html", "Jake VanderPlas", "Walks through broadcasting rules with shape diagrams and runnable examples.")
NP_BASICS = ("NumPy: the absolute basics for beginners", f"{NP}/user/absolute_beginners.html", "NumPy docs", "Covers array creation, reshaping, indexing and aggregation from scratch.")

GROUPS = [
    G(1, "dataframe dict", DSINTRO, HB_INDEX),
    G(2, "dataframe column", SUBSET, HB_INDEX),
    G(3, "loc iloc", INDEXING, HB_INDEX),
    G(4, "boolean dataframe", SUBSET, INDEXING),
    G(5, "index set_index", INDEXING, HB_INDEX),
    G(6, "nan dropna fillna", ("Working with missing data", f"{PD}/user_guide/missing_data.html", "pandas docs", "Official guide to detecting, dropping, filling and replacing missing values."), HB_MISSING),
    G(7, "duplicated keep", ("drop_duplicates() explained", "https://www.geeksforgeeks.org/python-pandas-dataframe-drop_duplicates/", "GeeksforGeeks", "Shows subset, keep and inplace on small DataFrames."), ("Duplicate Labels", f"{PD}/user_guide/duplicates.html", "pandas docs", "Explains duplicated rows and duplicate labels.")),
    G((8, 9), "dataframe replace rename", ("Essential basic functionality", f"{PD}/user_guide/basics.html", "pandas docs", "Covers renaming labels, astype and function application."), ("Working with missing data", f"{PD}/user_guide/missing_data.html", "pandas docs", "Includes the replace section with dict and regex forms.")),
    G(10, "dtype astype object category", ("Overview of Pandas Data Types", "https://pbpython.com/pandas_dtypes.html", "Practical Business Python", "Explains each pandas dtype and how astype conversions go wrong."), ("Essential basic functionality", f"{PD}/user_guide/basics.html", "pandas docs", "The dtypes and astype sections.")),
    G((11, 12), "groupby aggregate", GROUPBY, RP_GROUPBY),
    G(13, "pivot_table margins", ("Aggregation and Pivot Tables (Python Data Science Handbook)", f"{VP}/03.09-pivot-tables.html", "Jake VanderPlas", "Builds pivot tables as multi-dimensional groupby with the Titanic data."), ("Pandas Pivot Table Explained", "https://pbpython.com/pandas-pivot-table-explained.html", "Practical Business Python", "Explains aggfunc, margins and fill_value with a sales example.")),
    G(14, "crosstab", ("Pandas Crosstab Explained", "https://pbpython.com/pandas-crosstab.html", "Practical Business Python", "Explains crosstab counts, normalisation and margins, and how it differs from pivot_table."), RESHAPE),
    G(15, "groupby hierarchical multiindex", GROUPBY, ("Hierarchical Indexing (Python Data Science Handbook)", f"{VP}/03.05-hierarchical-indexing.html", "Jake VanderPlas", "Explains MultiIndex objects that multi-key groupbys produce.")),
    G(16, "groupby transform", GROUPBY, RP_GROUPBY),
    G(17, "merge join", MERGING, ("Combining Datasets: Merge and Join (Python Data Science Handbook)", f"{VP}/03.07-merge-and-join.html", "Jake VanderPlas", "Explains one-to-one, many-to-one and many-to-many joins on small tables.")),
    G(18, "concat axis", MERGING, ("Combining Datasets: Concat and Append (Python Data Science Handbook)", f"{VP}/03.06-concat-and-append.html", "Jake VanderPlas", "Explains concat along each axis and how indices behave.")),
    G((19, 20), "melt stack unstack pivot", RESHAPE, ("Hierarchical Indexing (Python Data Science Handbook)", f"{VP}/03.05-hierarchical-indexing.html", "Jake VanderPlas", "Shows stack/unstack between wide and long form.")),
    G(21, "apply lambda axis function", ("Essential basic functionality", f"{PD}/user_guide/basics.html", "pandas docs", "The function-application section: apply, map and pipe."), ("Pandas .apply() Explained", "https://realpython.com/pandas-apply/", "Real Python", "Explains apply vs vectorised operations and when each is faster.")),
    G((22, 24), "array arange reshape", NP_BASICS, ("The Basics of NumPy Arrays (Python Data Science Handbook)", f"{VP}/02.02-the-basics-of-numpy-arrays.html", "Jake VanderPlas", "Array attributes, indexing, slicing and reshaping with examples.")),
    G(23, "random normal", ("Random sampling (numpy.random)", f"{NP}/reference/random/index.html", "NumPy docs", "Explains the Generator API and seeding."), ("NumPy random.normal() Explained", "https://realpython.com/numpy-random-normal/", "Real Python", "Shows how to draw and check normal samples.")),
    G(25, "fancy indexing", ("Fancy Indexing (Python Data Science Handbook)", f"{VP}/02.07-fancy-indexing.html", "Jake VanderPlas", "Explains passing index arrays and combining them with masks."), ("Indexing on ndarrays", f"{NP}/user/basics.indexing.html", "NumPy docs", "Reference on advanced integer and boolean indexing.")),
    G(26, "boolean mask", ("Comparisons, Masks, and Boolean Logic (Python Data Science Handbook)", f"{VP}/02.06-boolean-arrays-and-masks.html", "Jake VanderPlas", "Builds boolean masks from comparisons and uses them to select data."), ("Indexing on ndarrays", f"{NP}/user/basics.indexing.html", "NumPy docs", "Boolean array indexing reference.")),
    G(27, "copy view array", ("Copies and views", f"{NP}/user/basics.copies.html", "NumPy docs", "Explains when slicing returns a view and when an operation copies."), ("The Basics of NumPy Arrays (Python Data Science Handbook)", f"{VP}/02.02-the-basics-of-numpy-arrays.html", "Jake VanderPlas", "Array slices as views, with a mutation example.")),
    G((28, 29, 30, 31, 32, 35, 37), "broadcasting shape", BROADCAST, HB_BROADCAST),
    G((33, 36), "aggregate min max sum axis", ("Aggregations: Min, Max, and Everything In Between (Python Data Science Handbook)", f"{VP}/02.04-computation-on-arrays-aggregates.html", "Jake VanderPlas", "Covers sum, min, max, mean and the axis argument."), NP_BASICS),
    G((34, 39), "sort argsort searchsorted", ("Sorting Arrays (Python Data Science Handbook)", f"{VP}/02.08-sorting.html", "Jake VanderPlas", "Covers sort, argsort, partition and sorting along an axis."), ("Fancy Indexing (Python Data Science Handbook)", f"{VP}/02.07-fancy-indexing.html", "Jake VanderPlas", "Includes binning data with searchsorted and counting with add.at.")),
    G(38, "bincount scatter add", ("Fancy Indexing (Python Data Science Handbook)", f"{VP}/02.07-fancy-indexing.html", "Jake VanderPlas", "Shows np.add.at and histogram-by-binning, the scatter-add pattern."), ("numpy.bincount", f"{NP}/reference/generated/numpy.bincount.html", "NumPy docs", "Reference for counting integer occurrences with optional weights.")),
]
