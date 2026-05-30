# categories.py — benchmark category constants
# Mirrors ZLinq's sandbox/Benchmark/Categories.cs


class From:
    List        = "FromList"
    NumpyF64    = "FromNumpyF64"
    NumpyI64    = "FromNumpyI64"
    CSV         = "FromCSV"
    JsonLines   = "FromJsonLines"
    Generator   = "FromGenerator"


class Methods:
    # Lazy combinators
    Filter      = "Filter"
    Map         = "Map"
    Take        = "Take"
    Skip        = "Skip"
    TakeWhile   = "TakeWhile"
    SkipWhile   = "SkipWhile"
    Enumerate   = "Enumerate"
    Chain       = "Chain"

    # Terminal / sinks
    ToList      = "ToList"
    Count       = "Count"
    Sum         = "Sum"
    Min         = "Min"
    Max         = "Max"
    First       = "First"
    Last        = "Last"
    Any         = "Any"
    All         = "All"
    Reduce      = "Reduce"
    ForEach     = "ForEach"


class Baseline:
    PythonListComp  = "Python-ListComprehension"
    PythonGenerator = "Python-Generator"
    Numpy           = "Numpy"
    Pandas          = "Pandas"
    Itertools       = "Itertools"


class Tags:
    SIMD     = "simd"
    Parallel = "parallel"
    GILFree  = "gil-free"
    DSL      = "dsl"
    Lambda   = "lambda"
    SmallN   = "small-n"
    LargeN   = "large-n"
