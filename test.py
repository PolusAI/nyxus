from nyxus import Nyxus 


nyx = Nyxus(["FOCUS_SCORE"])

features = nyx.featurize_directory("/Users/mckinziejr/Documents/GitHub/nyxus/data/int", "/Users/mckinziejr/Documents/GitHub/nyxus/data/int")

print(features)