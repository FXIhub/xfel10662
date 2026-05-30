from damnit import Damnit
import pandas as pd
import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
df = Damnit(os.environ['SANDBOX']).table(with_titles=True)
print(df)
