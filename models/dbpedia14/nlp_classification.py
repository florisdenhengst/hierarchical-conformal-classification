from datasets import load_dataset

dbpedia = load_dataset('fancyzhx/dbpedia_14')
print(dbpedia.label_names)
